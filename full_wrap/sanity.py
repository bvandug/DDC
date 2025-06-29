# hyperparameter_tuning.py
import logging, optuna                       # DEBUG logs from Optuna
optuna.logging.set_verbosity(logging.DEBUG)

import os, sys, json, torch, numpy as np
from torch import nn
from tqdm import tqdm
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG, DQN
from stable_baselines3.common.noise import NormalActionNoise
from simulink_env import SimulinkEnv, DiscretizedActionWrapper

# ------------------------------------------------------------------ #
# CONSTANTS (quick 1 k-step smoke-test)                              #
TOTAL_TIMESTEPS   = 1_000
EVAL_INTERVAL     =   100
N_EVAL_EPISODES   =     1
HARD_FAIL_THRESHOLDS = {300: 5, 600: 25, 900: 50}
SOLVED_THRESHOLD  = 150

MIN_RESOURCES    = 300   # first SH rung
REDUCTION_FACTOR = 2

TB_ROOT = "./hp_logs"    # TensorBoard root
os.makedirs(TB_ROOT, exist_ok=True)
# ------------------------------------------------------------------ #

def log(algo: str, tid: int, msg: str):
    print(f"[{algo.upper()}|Trial {tid}] {msg}", flush=True)

# ================================================================== #
def objective(trial, algo_name):
    base_env = SimulinkEnv(model_name="PendCart",
                           agent_block="PendCart/RL Agent",
                           dt=0.01)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(algo_name, trial.number, f"Device = {device}")

        tb_path = os.path.join(TB_ROOT,       # each trial gets its own dir
                               f"{algo_name}_T{trial.number}")

        # -------------------------------------------------------------- #
        # BUILD MODEL (example shows DQN; others identical idea)         #
        # -------------------------------------------------------------- #
        if algo_name == "dqn":
            forces = np.linspace(-10, 10, 11, dtype=np.float32)
            env    = DiscretizedActionWrapper(base_env, forces)

            p = trial
            params = dict(
                learning_rate=p.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                buffer_size=p.suggest_int("buffer_size", 50_000, 200_000),
                batch_size=p.suggest_int("batch_size", 32, 512),
                gamma=p.suggest_float("gamma", 0.90, 0.9999),
                tau=p.suggest_float("tau", 0.005, 0.05),
                exploration_fraction=p.suggest_float("exploration_fraction", 0.05, 0.4),
                exploration_final_eps=p.suggest_float("exploration_final_eps", 0.01, 0.1),
                target_update_interval=p.suggest_int("target_update_interval", 500, 10_000),
                train_freq=p.suggest_int("train_freq", 1, 8),
                n_layers=p.suggest_int("n_layers", 1, 3),
                layer_size=p.suggest_int("layer_size", 32, 256),
                activation_fn=p.suggest_categorical("activation_fn",
                                                    ["tanh", "relu", "leaky_relu", "elu"]),
            )
            net_arch = [params["layer_size"]]*params["n_layers"]
            act_map  = {"tanh": nn.Tanh, "relu": nn.ReLU,
                        "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}

            model = DQN(
                "MlpPolicy", env, device=device, verbose=0,
                tensorboard_log=tb_path,                # <-- writer created here
                **{k: params[k] for k in ["learning_rate", "buffer_size", "batch_size",
                                          "gamma", "tau",
                                          "exploration_fraction", "exploration_final_eps",
                                          "target_update_interval", "train_freq"]},
                policy_kwargs=dict(net_arch=net_arch,
                                   activation_fn=act_map[params["activation_fn"]]),
            )
            env_eval = env
        else:
            raise ValueError("For this demo only 'dqn' is wired.")  # keep short

        # -------------------------------------------------------------- #
        # TRAIN–EVAL–PRUNE LOOP                                         #
        # -------------------------------------------------------------- #
        timesteps = 0
        best = -np.inf
        while timesteps < TOTAL_TIMESTEPS:
            model.learn(EVAL_INTERVAL, reset_num_timesteps=False,
                        tb_log_name=f"T{trial.number}", progress_bar=False)
            timesteps += EVAL_INTERVAL

            # quick eval
            rewards = []
            for _ in range(N_EVAL_EPISODES):
                obs, done, ep_rew = env_eval.reset(), False, 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, done, _ = env_eval.step(action)
                    ep_rew += r
                rewards.append(ep_rew)
            mean_r = float(np.mean(rewards))
            best   = max(best, mean_r)
            trial.report(mean_r, timesteps)

            # hard-fail gates
            if (thr := HARD_FAIL_THRESHOLDS.get(timesteps)) is not None and mean_r < thr:
                log(algo_name, trial.number,
                    f"✘ Hard-fail {timesteps} (<{thr}, r={mean_r:.1f})")
                raise optuna.TrialPruned()

            # early success
            if mean_r >= SOLVED_THRESHOLD:
                log(algo_name, trial.number,
                    f"✔ Solved {timesteps} (r={mean_r:.1f})")
                break

            # SH pruner
            if trial.should_prune():
                log(algo_name, trial.number,
                    f"✘ Pruned by SH {timesteps} (r={mean_r:.1f})")
                raise optuna.TrialPruned()

        return best
    finally:
        base_env.close()

# ================================================================== #
def tune_hyperparameters(algo="dqn", n_trials=10, n_parallel=2):
    print(f"\nTuning {algo.upper()} with {n_parallel} workers")

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=MIN_RESOURCES, reduction_factor=REDUCTION_FACTOR)

    study = optuna.create_study(direction="maximize", pruner=pruner)
    pbar  = tqdm(total=n_trials, desc=f"{algo.upper()}",
                 dynamic_ncols=True, file=sys.stdout)

    def cb(study_, frozen):
        pbar.update(1)
        try:
            pbar.set_postfix(best=f"{study_.best_value:.1f}")
        except ValueError:
            pbar.set_postfix(best="–")

    study.optimize(lambda t: objective(t, algo),
                   n_trials=n_trials, n_jobs=n_parallel,
                   callbacks=[cb])

    pbar.close()
    print("Best reward:", study.best_value)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Smoke-test …")
    tune_hyperparameters("dqn", n_trials=6, n_parallel=2)
