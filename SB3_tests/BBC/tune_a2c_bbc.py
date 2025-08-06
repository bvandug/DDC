import os
import json
import sys
import random
import numpy as np
import optuna
import torch
from torch import nn
from tqdm import tqdm
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from np_bbc_env import JAXBuckBoostConverterEnv

# === Logging & tuning config ===
TB_ROOT           = "./jax_hp_logs"
STUDY_NAME        = "bbc_hp_optuna_a2c"
DB_URL            = f"sqlite:///{STUDY_NAME}.db"
RESULTS_ROOT      = "./jax_hp_results"
TOTAL_TIMESTEPS   = 100_000
EVAL_INTERVAL     = 10_000
N_EVAL_EPISODES   = 10
HARD_FAIL_THRESH  = {20_000: -120_000, 30_000: -80_000}
os.makedirs(TB_ROOT,      exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)

def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_env():
    return DummyVecEnv([lambda: Monitor(
        JAXBuckBoostConverterEnv(
            max_episode_steps=2000,
            frame_skip=10,
            grace_period_steps=50,
            dt=5e-6,
            target_voltage=-30.0
        )
    )])

def unwrap_reset(reset_out):
    # support both Gym v0.26+ and older
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0]
    return reset_out

def objective(trial: optuna.Trial) -> float:
    set_global_seeds(42)
    env    = make_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sample A2C hyperparameters
    lr            = trial.suggest_float("learning_rate", 1e-5,   1e-3, log=True)
    gamma         = trial.suggest_float("gamma",          0.90,  0.9999)
    n_steps       = trial.suggest_int("n_steps",           8,    2048, log=True)
    ent_coef      = trial.suggest_float("ent_coef",      1e-7,    0.1, log=True)
    vf_coef       = trial.suggest_float("vf_coef",        0.1,    1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm",   0.3,    5.0)
    gae_lambda    = trial.suggest_float("gae_lambda",      0.8,    1.0)
    rms_eps       = trial.suggest_float("rms_prop_eps", 1e-6,    1e-3, log=True)
    use_rms       = trial.suggest_categorical("use_rms_prop", [True, False])

    # architecture search
    activation_map = {
        "relu":       nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu":        nn.ELU,
        "selu":       nn.SELU,
        "silu":       nn.SiLU,
    }
    layer_size = trial.suggest_int("layer_size", 32, 512, log=True)
    n_layers   = trial.suggest_int("n_layers", 1, 6)
    act_fn     = activation_map[trial.suggest_categorical("activation_fn", list(activation_map.keys()))]

    policy_kwargs = {
        "net_arch":      [layer_size] * n_layers,
        "activation_fn": act_fn
    }

    model = A2C(
        "MlpPolicy", env,
        learning_rate=lr,
        gamma=gamma,
        n_steps=n_steps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        gae_lambda=gae_lambda,
        rms_prop_eps=rms_eps,
        use_rms_prop=use_rms,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(TB_ROOT, "a2c"),
        verbose=0,
        device=device
    )

    timesteps = 0
    while timesteps < TOTAL_TIMESTEPS:
        model.learn(
            EVAL_INTERVAL,
            reset_num_timesteps=False,
            progress_bar=False,
            tb_log_name=f"trial{trial.number}"
        )
        timesteps += EVAL_INTERVAL

        # evaluate
        rewards = []
        for _ in range(N_EVAL_EPISODES):
            obs = unwrap_reset(env.reset())
            done, ep_r = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _ = env.step(action)
                ep_r += float(r)
            rewards.append(ep_r)

        top8    = float(np.mean(sorted(rewards, reverse=True)[:8]))
        mean_all= float(np.mean(rewards))

        # log to TensorBoard
        model.logger.record("eval/top8_avg",   top8)
        model.logger.record("eval/mean_all10", mean_all)
        model.logger.dump(timesteps)

        # hard‐fail pruning
        if timesteps in HARD_FAIL_THRESH and top8 < HARD_FAIL_THRESH[timesteps]:
            print(f"[Trial {trial.number}] Pruned at {timesteps} steps: top8_avg={top8:.2f} < threshold {HARD_FAIL_THRESH[timesteps]}")
            raise optuna.TrialPruned(f"Hard‐fail at {timesteps}")

        # Optuna‐prune based on intermediate value
        trial.report(top8, step=timesteps)
        if trial.should_prune():
            print(f"[Trial {trial.number}] Pruned by Optuna at {timesteps} steps: intermediate top8_avg={top8:.2f}")
            raise optuna.TrialPruned(f"Underperforming at {timesteps}")

    # if we get here, trial completed
    print(f"[Trial {trial.number}] Completed: final top8_avg={top8:.2f}")
    env.close()
    return mean_all

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=DB_URL,
        load_if_exists=True
    )

    pbar = tqdm(total=50, desc="Tuning A2C (BBC)", dynamic_ncols=True)
    def _cb(study, trial):
        pbar.update(1)
        # show best known value so far
        try:
            bv = study.best_value
            pbar.set_postfix(best=f"{bv:.1f}")
        except ValueError:
            pbar.set_postfix(best="N/A")
        # print final status of this trial
        state = trial.state.name
        val   = trial.value if trial.value is not None else "None"
        print(f">>> Trial {trial.number} ended with status={state}, value={val}, params={trial.params}")
        sys.stdout.write("\n"); sys.stdout.flush()

    study.optimize(objective, n_trials=50, callbacks=[_cb], n_jobs=4)
    pbar.close()

    # Save best
    with open(os.path.join(RESULTS_ROOT, "a2c_best_params.json"), "w") as f:
        json.dump({
            "best_params": study.best_params,
            "best_value":  study.best_value
        }, f, indent=2)

    print(f"\nBest mean top8 reward: {study.best_value:.1f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
