# hyperparameter_tuning.py
import optuna
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG, DQN
from stable_baselines3.common.noise import NormalActionNoise
from simulink_env import SimulinkEnv, DiscretizedActionWrapper
import json, os, sys, torch
from torch import nn
from tqdm import tqdm

# --------------------------------------------------------------------- #
# CONSTANTS                                                             #
TOTAL_TIMESTEPS   = 50_000      # per trial during HP search
EVAL_INTERVAL     = 10_000       # evaluate & report every 10 k steps
N_EVAL_EPISODES   = 10            # quick roll-outs for pruning
SOLVED_THRESHOLD  = 490           # “good enough” reward, end trial early
# HARD_FAIL_THRESHOLDS = {10_000: 30, 20_000: 150, 30_000: 200, 40_000: 250}
HARD_FAIL_THRESHOLDS = {20_000: 20, 30_000: 50}


TB_ROOT = "./hp_logs"
# --------------------------------------------------------------------- #
os.makedirs(TB_ROOT, exist_ok=True)      # <<< 1. always create the folder
# --------------------------------------------------------------------- #

MIN_RESOURCES = 20000  # min. resources for pruning
REDUCTION_FACTOR = 2  # reduction factor for pruning
MIN_EARLY_STOPPING_RATE = 0  # min. early stopping rate for pruning
# --------------------------------------------------------------------- #



def objective(trial, algo_name):
    """
    One Optuna trial: build the agent, train in chunks, evaluate,
    report intermediate scores and allow pruning.
    """
    base_env = SimulinkEnv(model_name="PendCart",
                           agent_block="PendCart/RL Agent",
                           dt=0.01)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{algo_name.upper()}] Using device: {device}")

        # ================================================================= //
        # ------------  Build model & hyper-parameter search  ------------- //
        # ================================================================= #
        if algo_name == "td3":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size":  trial.suggest_int("buffer_size", 50_000, 200_000),
                "batch_size":   trial.suggest_int("batch_size", 64, 512),
                "tau":          trial.suggest_float("tau", 0.001, 0.02),
                "gamma":        trial.suggest_float("gamma", 0.9, 0.9999),
                "policy_delay": trial.suggest_int("policy_delay", 1, 4),
                "action_noise_sigma":  trial.suggest_float("action_noise_sigma", 0.1, 0.5),
                "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.5),
                "target_noise_clip":   trial.suggest_float("target_noise_clip", 0.3, 0.7),
                "n_layers":     trial.suggest_int("n_layers", 1, 3),
                "layer_size":   trial.suggest_int("layer_size", 32, 256),
                "activation_fn":trial.suggest_categorical("activation_fn",
                                     ["tanh", "relu", "leaky_relu", "elu"]),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU,
                              "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
            policy_kwargs = {"net_arch": net_arch,
                             "activation_fn": activation_map[params["activation_fn"]]}

            model = TD3(
                "MlpPolicy", base_env, device=device, verbose=0,
                learning_rate=params["learning_rate"],
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"], gamma=params["gamma"],
                train_freq=(1, "step"), policy_delay=params["policy_delay"],
                action_noise=NormalActionNoise(mean=np.zeros(1),
                                               sigma=params["action_noise_sigma"] * np.ones(1)),
                target_policy_noise=params["target_policy_noise"],
                target_noise_clip=params["target_noise_clip"],
                policy_kwargs=policy_kwargs,
            )
            env_for_eval = base_env

        # ------------------------------------------------------------------ #
        elif algo_name == "a2c":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
                "n_steps": trial.suggest_int("n_steps", 8, 2048, log=True),
                "ent_coef": trial.suggest_float("ent_coef", 1e-7, 0.1, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
                "rms_prop_eps": trial.suggest_float("rms_prop_eps", 1e-6, 1e-3, log=True),
                "use_rms_prop": trial.suggest_categorical("use_rms_prop", [True, False]),
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "layer_size": trial.suggest_int("layer_size", 32, 256),
                "activation_fn": trial.suggest_categorical(
                    "activation_fn", ["tanh", "relu", "leaky_relu", "elu"]),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU,
                              "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
            policy_kwargs = {"net_arch": net_arch,
                             "activation_fn": activation_map[params["activation_fn"]]}
            model = A2C(
                "MlpPolicy", base_env, device=device, verbose=0,
                learning_rate=params["learning_rate"],
                gamma=params["gamma"], n_steps=params["n_steps"],
                ent_coef=params["ent_coef"], vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                rms_prop_eps=params["rms_prop_eps"],
                use_rms_prop=params["use_rms_prop"],
                policy_kwargs=policy_kwargs,
            )
            env_for_eval = base_env

        # ------------------------------------------------------------------ #
        elif algo_name == "ppo":
            def valid_batch_sizes(n, lo=32, hi=512):
                return [i for i in range(lo, min(n+1, hi+1)) if n % i == 0]

            n_steps = trial.suggest_int("n_steps", 64, 2048, log=True)
            batch_size = np.random.choice(valid_batch_sizes(n_steps)) \
                         if valid_batch_sizes(n_steps) else n_steps
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "n_steps": n_steps, "batch_size": batch_size,
                "n_epochs": trial.suggest_int("n_epochs", 4, 20),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
                "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "layer_size": trial.suggest_int("layer_size", 32, 256),
                "activation_fn": trial.suggest_categorical(
                    "activation_fn", ["tanh", "relu", "leaky_relu", "elu"]),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU,
                              "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
            policy_kwargs = {"net_arch": net_arch,
                             "activation_fn": activation_map[params["activation_fn"]]}
            model = PPO(
                "MlpPolicy", base_env, device=device, verbose=0,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"], batch_size=params["batch_size"],
                n_epochs=params["n_epochs"], gamma=params["gamma"],
                clip_range=params["clip_range"], ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"], max_grad_norm=params["max_grad_norm"],
                gae_lambda=params["gae_lambda"], policy_kwargs=policy_kwargs,
            )
            env_for_eval = base_env

        # ------------------------------------------------------------------ #
        elif algo_name == "ddpg":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
                "batch_size": trial.suggest_int("batch_size", 64, 512),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "action_noise_sigma": trial.suggest_float("action_noise_sigma", 0.1, 0.5),
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "layer_size": trial.suggest_int("layer_size", 32, 256),
                "activation_fn": trial.suggest_categorical(
                    "activation_fn", ["tanh", "relu", "leaky_relu", "elu"]),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU,
                              "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
            policy_kwargs = {"net_arch": net_arch,
                             "activation_fn": activation_map[params["activation_fn"]]}
            model = DDPG(
                "MlpPolicy", base_env, device=device, verbose=0,
                learning_rate=params["learning_rate"],
                buffer_size=params["buffer_size"], batch_size=params["batch_size"],
                tau=params["tau"], gamma=params["gamma"],
                train_freq=(1, "step"),
                action_noise=NormalActionNoise(mean=np.zeros(1),
                                               sigma=params["action_noise_sigma"] * np.ones(1)),
                policy_kwargs=policy_kwargs,
            )
            env_for_eval = base_env

        # ------------------------------------------------------------------ #
        elif algo_name == "sac":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
                "batch_size": trial.suggest_int("batch_size", 64, 512),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.001, 0.01, 0.1]),
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "layer_size": trial.suggest_int("layer_size", 32, 256),
                "activation_fn": trial.suggest_categorical(
                    "activation_fn", ["tanh", "relu", "leaky_relu", "elu"]),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU,
                              "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
            policy_kwargs = {"net_arch": net_arch,
                             "activation_fn": activation_map[params["activation_fn"]]}
            model = SAC(
                "MlpPolicy", base_env, device=device, verbose=0,
                learning_rate=params["learning_rate"],
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"], gamma=params["gamma"],
                ent_coef=params["ent_coef"],
                policy_kwargs=policy_kwargs,
            )
            env_for_eval = base_env

        # ------------------------------------------------------------------ #
        elif algo_name == "dqn":
            force_values = np.linspace(-10, 10, 11, dtype=np.float32)
            wrapped_env  = DiscretizedActionWrapper(base_env, force_values)

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True),
                "buffer_size": trial.suggest_int("buffer_size", 50_000, 150_000),
                "batch_size": trial.suggest_int("batch_size", 32, 512, step = 32),
                "gamma": trial.suggest_float("gamma", 0.95, 0.995),
                "tau": trial.suggest_float("tau", 0.01, 0.04),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.15, 0.4),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
                "target_update_interval": trial.suggest_int("target_update_interval", 1_000, 5_000),
                "train_freq": trial.suggest_int("train_freq", 1, 6),
                "n_layers": trial.suggest_int("n_layers", 2, 6),
                "layer_size": trial.suggest_int("layer_size", 128, 512, step = 64),
                "activation_fn": trial.suggest_categorical(
                    "activation_fn", ["tanh", "relu", "leaky_relu", "elu"]),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU,
                              "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
            model = DQN(
                "MlpPolicy", wrapped_env, device=device, verbose=0,
                tensorboard_log= TB_ROOT,
                learning_rate=params["learning_rate"],
                buffer_size=params["buffer_size"], batch_size=params["batch_size"],
                gamma=params["gamma"], tau=params["tau"],
                exploration_fraction=params["exploration_fraction"],
                exploration_final_eps=params["exploration_final_eps"],
                target_update_interval=params["target_update_interval"],
                train_freq=params["train_freq"],
                policy_kwargs=dict(net_arch=net_arch,
                                   activation_fn=activation_map[params["activation_fn"]]),
                learning_starts=5000,  # start training after 1000 steps
            )
            env_for_eval = wrapped_env

        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        # ================================================================= //
        # -------------  TRAIN–EVAL LOOP WITH PRUNING  -------------------- //
        # ================================================================= #
        timesteps = 0
        best_mean = -np.inf

        while timesteps < TOTAL_TIMESTEPS:
            model.learn(EVAL_INTERVAL, reset_num_timesteps=False, progress_bar=False, tb_log_name=f"T{trial.number}")
            timesteps += EVAL_INTERVAL

            # quick deterministic eval
            rewards = []
            for _ in range(N_EVAL_EPISODES):
                obs, done, ep_rew = env_for_eval.reset(), False, 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = env_for_eval.step(action)
                    ep_rew += reward
                rewards.append(ep_rew)
            mean_reward = float(np.mean(rewards))
            best_mean   = max(best_mean, mean_reward)
            trial.report(mean_reward, timesteps)

            # # ---------- 1) Hard-fail gates ----------
            # if timesteps in HARD_FAIL_THRESHOLDS:
            #     threshold = HARD_FAIL_THRESHOLDS[timesteps]
            #     if mean_reward < threshold:
            #         print(f"[{algo_name.upper()}|Trial {trial.number}] ✘ "
            #               f"Hard-fail (<{threshold}) at {timesteps} "
            #               f"(reward {mean_reward:.1f})", flush=True)
            #         raise optuna.TrialPruned()

            # ---------- 2) Early success ----------
            if mean_reward >= SOLVED_THRESHOLD:
                print(f"[{algo_name.upper()}|Trial {trial.number}] ✔ "
                      f"Solved at {timesteps} (reward {mean_reward:.1f})", flush=True)
                break

            # ---------- 3) Successive-Halving pruner ----------
            if trial.should_prune():
                print(f"[{algo_name.upper()}|Trial {trial.number}] ✘ "
                      f"Pruned by Halving at {timesteps} "
                      f"(reward {mean_reward:.1f})", flush=True)
                raise optuna.TrialPruned()

        return best_mean

    finally:
        base_env.close()

# --------------------------------------------------------------------------- #
def tune_hyperparameters(algo_name, n_trials=50, n_parallel=6):
    print(f"\nTuning {algo_name.upper()} with {n_parallel} parallel workers")

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=MIN_RESOURCES, reduction_factor=REDUCTION_FACTOR, min_early_stopping_rate=MIN_EARLY_STOPPING_RATE)

    study = optuna.create_study(direction="maximize", pruner=pruner)

    # ---------- enqueue good baseline params ----------
    if algo_name == "dqn":
        study.enqueue_trial({
            "learning_rate": 5e-4, "buffer_size": 100_000, "batch_size": 128,
            "gamma": 0.99, "tau": 0.01, "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05, "target_update_interval": 1_000,
            "train_freq": 4, "n_layers": 2, "layer_size": 128,
            "activation_fn": "relu",
        })
    elif algo_name == "td3":
        study.enqueue_trial({
            "learning_rate": 9.53e-4, "buffer_size": 127_040, "batch_size": 511,
            "tau": 0.0077, "gamma": 0.943, "policy_delay": 4,
            "action_noise_sigma": 0.102, "target_policy_noise": 0.105,
            "target_noise_clip": 0.564, "n_layers": 2, "layer_size": 64,
            "activation_fn": "relu",
        })
    elif algo_name == "ppo":
        study.enqueue_trial({
            "n_steps": 68, "batch_size": 34, "learning_rate": 3.45e-4,
            "n_epochs": 4, "gamma": 0.948, "clip_range": 0.132,
            "ent_coef": 1.1e-5, "vf_coef": 0.625, "max_grad_norm": 2.499,
            "gae_lambda": 0.938, "n_layers": 3, "layer_size": 249,
            "activation_fn": "tanh",
        })

    pbar = tqdm(total=n_trials, desc=f"Tuning {algo_name.upper()}",
                file=sys.stdout, dynamic_ncols=True, leave=True)

    def _cb(st, tr):
        pbar.update(1)
        try:
            best_val = st.best_value        # only exists if ≥1 COMPLETE trial
            pbar.set_postfix(best_val=f"{best_val:.1f}")
        except ValueError:                  # no completed trials yet
            pbar.set_postfix(best_val="–")

    study.optimize(lambda t: objective(t, algo_name),
                   n_trials=n_trials, n_jobs=n_parallel,
                   callbacks=[_cb])

    pbar.close()

    # -------- save results --------
    os.makedirs("hyperparameter_results", exist_ok=True)
    with open(f"hyperparameter_results/{algo_name}_best_params.json", "w") as fh:
        json.dump({
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials":   n_trials,
        }, fh, indent=4)

    # print summary
    print(f"\nBest for {algo_name.upper()}: reward={study.best_value:.2f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    return study.best_params

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    algorithms = ["dqn"]           # change this list to tune other algos
    print("Starting hyperparameter tuning …")
    for algo in algorithms:
        print("\n" + "="*60)
        print(f"Tuning {algo.upper()} …")
        print("="*60 + "\n")
        tune_hyperparameters(algo, n_trials=50, n_parallel=6)
