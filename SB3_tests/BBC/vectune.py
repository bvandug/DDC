# bbc_a2c_hp.py — Optuna tuning for A2C on JAXBuckBoostConverterEnv (2M steps/trial)
import os, sys, json, random, inspect
import numpy as np
import torch
import optuna
from torch import nn
from tqdm import tqdm

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from np_bbc_env import JAXBuckBoostConverterEnv

# ===== Config =====
TB_ROOT            = "./bbc_hp_logs"
RESULTS_DIR        = "./bbc_hp_results"
TOTAL_TIMESTEPS    = 2_000_000
EVAL_INTERVAL      = 100_000
LAST_EVALS_AVG     = 5
N_EVAL_EPISODES    = 10
N_ENVS             = 8
SEED               = 42

# ---- Optuna DB (save inside RESULTS_DIR) ----
DB_NAME = "bbc_a2c_optuna.db"
DB_PATH = os.path.join(RESULTS_DIR, DB_NAME)
STORAGE_URL = f"sqlite:///{DB_PATH.replace(os.sep, '/')}"

os.makedirs(TB_ROOT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_global_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(rank: int = 0):
    def _thunk():
        e = JAXBuckBoostConverterEnv(
            dt=5e-6,
            frame_skip=10,           # MATCH training script (20 kHz)
            max_episode_steps=4000,  # MATCH training script
            grace_period_steps=100,
            target_voltage=-30.0,
            enforce_dcm=True,
        )

        # Same Monitor fields as training (harmless for tuning; useful if you log later)
        e = Monitor(
            e,
            info_keywords=("iL", "vC", "mag_vC", "e_norm", "dduty", "in_band"),
        )
        e.reset(seed=SEED + rank)   # Gymnasium-style seeding
        return e
    return _thunk


def make_vec_env():
    fns = [make_env(i) for i in range(N_ENVS)]
    vec_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
    env = vec_cls(fns)
    # MATCH training: normalize observations only; rewards stay raw
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_reward=10.0)
    return env


def clone_eval_env_from(train_env: VecNormalize):
    """
    Separate eval env with frozen stats and RAW rewards.
    We copy obs_rms from the training env before each eval.
    """
    fns = [make_env(10_000 + i) for i in range(N_ENVS)]
    vec_cls = SubprocVecEnv if N_ENVS > 1 else DummyVecEnv
    eval_env = vec_cls(fns)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_reward=10.0)
    eval_env.training = False  # freeze stats
    eval_env.reset()
    eval_env.obs_rms = train_env.obs_rms  # initial sync; refreshed before each eval
    return eval_env


def learn_supports_progress_bar(model) -> bool:
    return "progress_bar" in inspect.signature(model.learn).parameters


def eval_policy_raw(model, venv: VecNormalize, n_episodes=10):
    """Deterministic eval on RAW rewards using frozen VecNormalize eval env."""
    returns = []
    obs = venv.reset()
    while len(returns) < n_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(actions)
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                returns.append(float(ep["r"]))
                if len(returns) >= n_episodes:
                    break
    return returns[:n_episodes]


# ===== Parametric architecture search (no presets) =====
def a2c_policy_kwargs_from_trial(trial: optuna.Trial):
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
    activation = acts[trial.suggest_categorical("activation_fn", list(acts.keys()))]

    def qround(x, q=32, lo=64, hi=512):
        return int(min(hi, max(lo, round(x / q) * q)))

    n_layers = trial.suggest_int("n_layers", 1, 4)
    base_log2 = trial.suggest_float("log2_base_width", 6.0, 9.0)   # 64..512
    base = qround(2 ** base_log2)

    shape = trial.suggest_categorical("shape", ["flat", "taper", "pyramid", "reverse_pyramid"])
    tie_pi_vf = trial.suggest_categorical("tie_pi_vf", [True, False])
    pi_scale  = trial.suggest_float("pi_scale", 0.75, 1.5)
    vf_scale  = pi_scale if tie_pi_vf else trial.suggest_float("vf_scale", 0.75, 1.5)

    def make_shape(base_width, layers, kind, scale):
        if kind == "flat":
            widths = [base_width] * layers
        elif kind == "taper":
            widths = [qround(base_width * (0.8 ** i)) for i in range(layers)]
        elif kind == "pyramid":
            widths = [qround(base_width * (1.15 ** i)) for i in range(layers)]
        else:  # reverse_pyramid
            widths = [qround(base_width * (1.15 ** (layers - 1 - i))) for i in range(layers)]
        return [qround(w * scale) for w in widths]

    pi_arch = make_shape(base, n_layers, shape, pi_scale)
    vf_arch = pi_arch if tie_pi_vf else make_shape(base, n_layers, shape, vf_scale)

    return dict(
        net_arch=dict(pi=pi_arch, vf=vf_arch),  # SB3 1.8+: dict only (no shared trunk)
        activation_fn=activation,
        log_std_init=trial.suggest_float("log_std_init", -2.5, -0.5),
        ortho_init=trial.suggest_categorical("ortho_init", [False, True]),
    )


def objective(trial: optuna.Trial):
    set_global_seeds(SEED)

    env = make_vec_env()
    eval_env = clone_eval_env_from(env)

    device = "cpu"  # A2C + MLP often better on CPU

    # --- HP search space (A2C only) ---
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        "n_steps": trial.suggest_categorical("n_steps", [32, 64, 128, 256]),
        "ent_coef": trial.suggest_float("ent_coef", 1e-5, 0.02, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        "rms_prop_eps": trial.suggest_float("rms_prop_eps", 1e-6, 1e-4, log=True),
        "use_rms_prop": trial.suggest_categorical("use_rms_prop", [True, False]),
    }
    policy_kwargs = a2c_policy_kwargs_from_trial(trial)

    model = A2C(
        "MlpPolicy",
        env,
        device=device,
        verbose=0,
        tensorboard_log=os.path.join(TB_ROOT, "a2c"),
        policy_kwargs=policy_kwargs,
        **params,
    )

    timesteps = 0
    eval_history = []
    best_so_far = -np.inf
    best_dir = None
    logged_bs = False  # log effective batch size once after first learn()

    try:
        while timesteps < TOTAL_TIMESTEPS:
            learn_kwargs = dict(reset_num_timesteps=False, tb_log_name=f"T{trial.number}")
            if learn_supports_progress_bar(model):
                learn_kwargs["progress_bar"] = False

            model.learn(EVAL_INTERVAL, **learn_kwargs)
            timesteps += EVAL_INTERVAL

            # log effective batch size once (logger now exists)
            if not logged_bs:
                try:
                    model.logger.record("train/effective_batch_size", params["n_steps"] * N_ENVS)
                    model.logger.dump(timesteps)
                except Exception:
                    pass
                logged_bs = True

            # Keep eval obs stats in sync with training env before each eval
            eval_env.obs_rms = env.obs_rms

            rewards = eval_policy_raw(model, eval_env, n_episodes=N_EVAL_EPISODES)
            if not np.isfinite(rewards).all():
                raise optuna.TrialPruned()

            top_k = min(8, len(rewards))
            top_k_avg = float(np.mean(sorted(rewards, reverse=True)[:top_k]))
            eval_history.append(top_k_avg)

            # Log to TB
            model.logger.record("eval/topk_avg", top_k_avg)
            model.logger.record("eval/mean_all", float(np.mean(rewards)))
            model.logger.record("tune/timesteps", timesteps)
            model.logger.dump(timesteps)

            # hard-floor pruning for clearly bad configs
            if timesteps >= 300_000 and top_k_avg < -5e4:
                raise optuna.TrialPruned()

            # save best artifacts for this trial
            if top_k_avg > best_so_far:
                best_so_far = top_k_avg
                save_dir = os.path.join(RESULTS_DIR, f"a2c_trial{trial.number}_best")
                os.makedirs(save_dir, exist_ok=True)
                model.save(os.path.join(save_dir, "model"))
                env.save(os.path.join(save_dir, "vecnorm.pkl"))
                with open(os.path.join(save_dir, "meta.json"), "w") as f:
                    json.dump(
                        {
                            "trial": trial.number,
                            "timesteps": timesteps,
                            "topk_avg": top_k_avg,
                            "params": params,
                            "policy_kwargs": {k: str(v) for k, v in policy_kwargs.items()},
                        },
                        f,
                        indent=2,
                    )
                best_dir = save_dir

            trial.set_user_attr("best_dir", best_dir)
            trial.set_user_attr("best_score", best_so_far)
            trial.report(top_k_avg, step=timesteps)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Objective: mean of last K evals to prefer configs that age well
        final_score = float(np.mean(eval_history[-LAST_EVALS_AVG:]))
        return final_score

    finally:
        try: env.close()
        except Exception: pass
        try: eval_env.close()
        except Exception: pass


def tune_a2c(n_trials=20, n_parallel=1):
    print("Tuning A2C on Buck-Boost… (2M steps/trial)")
    print(f"[Optuna] Using SQLite storage at: {DB_PATH}")

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=EVAL_INTERVAL * 2,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        study_name="bbc_a2c_tuning_2M",
        storage=STORAGE_URL,         # <-- moved into RESULTS_DIR
        load_if_exists=True,
    )

    pbar = tqdm(total=n_trials, desc="Tuning A2C", file=sys.stdout, dynamic_ncols=True)

    def _cb(study_, trial_):
        pbar.update(1)
        try:
            pbar.set_postfix(best_val=f"{study_.best_value:.1f}")
        except Exception:
            pbar.set_postfix(best_val="–")
        sys.stdout.write("\n")
        sys.stdout.flush()

    study.optimize(objective, n_trials=n_trials, n_jobs=n_parallel, callbacks=[_cb])
    pbar.close()

    best_trial = study.best_trial
    best_dir = best_trial.user_attrs.get("best_dir", None)

    out = {
        "best_params": best_trial.params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "best_trial_number": best_trial.number,
        "best_artifacts_dir": best_dir,
        "notes": f"Final score is mean of last {LAST_EVALS_AVG} evals (top-k avg each).",
    }
    with open(os.path.join(RESULTS_DIR, "a2c_best_params.json"), "w") as f:
        json.dump(out, f, indent=4)

    print(f"Best (mean of last {LAST_EVALS_AVG} evals): {study.best_value:.2f}")
    for k, v in best_trial.params.items():
        print(f"  {k}: {v}")
    if best_dir:
        print(f"Best artifacts saved in: {best_dir}")
    return best_trial.params


if __name__ == "__main__":
    # Keep the spawn guard on Windows if you increase n_jobs
    best = tune_a2c(n_trials=50, n_parallel=1)
