import optuna
import sys
import numpy as np
import os, json, sys, torch, random
from torch import nn
from tqdm import tqdm
from stable_baselines3 import PPO, A2C, TD3, DDPG, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from cp_jax_wrapper import CartPoleGymWrapper, DiscretizedActionWrapper

TB_ROOT = "./jax_hp_logs"
TOTAL_TIMESTEPS = 50_000
EVAL_INTERVAL = 10_000
N_EVAL_EPISODES = 10
HARD_FAIL_THRESHOLDS = {20_000: 20, 30_000: 50}
MIN_RESOURCES = 20000
REDUCTION_FACTOR = 2
MIN_EARLY_STOPPING_RATE = 0
os.makedirs(TB_ROOT, exist_ok=True)

def set_global_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def make_env(algo_name):
    base = CartPoleGymWrapper(seed=42)
    if algo_name == "dqn":
        return DummyVecEnv([lambda: Monitor(DiscretizedActionWrapper(base, [-10.0, 0.0, 10.0]))])
    return DummyVecEnv([lambda: Monitor(base)])

def objective(trial, algo_name):
    set_global_seeds(42)
    env = make_env(algo_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}

    def policy_kwargs_from_trial(trial):
        return {
            "net_arch": [trial.suggest_int("layer_size", 32, 512)] * trial.suggest_int("n_layers", 1, 4),
            "activation_fn": activation_map[trial.suggest_categorical("activation_fn", list(activation_map.keys()))]
        }

    if algo_name == "ppo":
        n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        if batch_size > n_steps or n_steps % batch_size != 0:
            print(f"Pruning trial {trial.number}: batch_size={batch_size} incompatible with n_steps={n_steps}", flush=True)
            raise optuna.TrialPruned()
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True),
            "n_steps": n_steps, "batch_size": batch_size,
            "n_epochs": trial.suggest_int("n_epochs", 4, 20),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        }
        model = PPO("MlpPolicy", env, device=device, verbose=0, tensorboard_log=os.path.join(TB_ROOT, algo_name),
                    policy_kwargs=policy_kwargs_from_trial(trial), **params)

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
        }
        model = A2C("MlpPolicy", env, device=device, verbose=0, tensorboard_log=os.path.join(TB_ROOT, algo_name),
                    policy_kwargs=policy_kwargs_from_trial(trial), **params)

    elif algo_name in ["td3", "ddpg"]:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
            "batch_size": trial.suggest_int("batch_size", 64, 512),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "train_freq": (1, "episode"),
            "gradient_steps": -1,
        }
        action_noise = NormalActionNoise(mean=np.zeros(1), sigma=trial.suggest_float("action_noise_sigma", 0.1, 0.5) * np.ones(1))
        model_class = TD3 if algo_name == "td3" else DDPG
        model = model_class("MlpPolicy", env, device=device, verbose=0, tensorboard_log=os.path.join(TB_ROOT, algo_name),
                    policy_kwargs=policy_kwargs_from_trial(trial), **params)

    elif algo_name == "sac":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
            "batch_size": trial.suggest_int("batch_size", 64, 512),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.001, 0.01, 0.1]),
        }
        model = SAC("MlpPolicy", env, device=device, verbose=0, tensorboard_log=os.path.join(TB_ROOT, algo_name),
                    policy_kwargs=policy_kwargs_from_trial(trial), **params)

    elif algo_name == "dqn":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 150_000),
            "batch_size": trial.suggest_int("batch_size", 32, 512, step=32),
            "gamma": trial.suggest_float("gamma", 0.95, 0.995),
            "tau": trial.suggest_float("tau", 0.01, 0.03),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.15, 0.4),
            "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
            "target_update_interval": trial.suggest_int("target_update_interval", 1_000, 5_000),
            "train_freq": trial.suggest_int("train_freq", 1, 6),
        }
        model = DQN("MlpPolicy", env, device=device, verbose=0,tensorboard_log=os.path.join(TB_ROOT, algo_name),
                    learning_starts=5000,
                    policy_kwargs=policy_kwargs_from_trial(trial), **params)

    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

    timesteps = 0
    while timesteps < TOTAL_TIMESTEPS:
        model.learn(EVAL_INTERVAL, reset_num_timesteps=False, progress_bar=False, tb_log_name=f"T{trial.number}")
        timesteps += EVAL_INTERVAL

        rewards = []
        for _ in range(N_EVAL_EPISODES):
            obs = env.reset()
            done, ep_rew = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                ep_rew += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            rewards.append(ep_rew)

        top_k_avg = float(np.mean(sorted(rewards, reverse=True)[:8]))
        model.logger.record("eval/top8_avg", top_k_avg)
        model.logger.record("eval/mean_all10", float(np.mean(rewards)))
        model.logger.dump(timesteps)

        if timesteps in HARD_FAIL_THRESHOLDS and top_k_avg < HARD_FAIL_THRESHOLDS[timesteps]:
            print(f"Pruning trial {trial.number}: reward {top_k_avg:.2f} below threshold {HARD_FAIL_THRESHOLDS[timesteps]} at {timesteps} steps", flush=True)
            raise optuna.TrialPruned()

        trial.report(top_k_avg, step=timesteps)
        if trial.should_prune():
            print(f"Pruning trial {trial.number}: Optuna judged it underperforming at step {timesteps} with reward {top_k_avg:.2f}", flush=True)
            raise optuna.TrialPruned()

    return top_k_avg

def tune_hyperparameters(algo_name, n_trials=50, n_parallel=4):
    print(f"Tuning {algo_name.upper()} on JAX env with SB3...")
    storage_path = f"sqlite:///jax_optuna_{algo_name}.db"
    study = optuna.create_study(direction="maximize",
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=MIN_RESOURCES,
            reduction_factor=REDUCTION_FACTOR,
            min_early_stopping_rate=MIN_EARLY_STOPPING_RATE),
        study_name=f"jax_{algo_name}_tuning",
        storage=storage_path,
        load_if_exists=True)

    pbar = tqdm(total=n_trials, desc=f"Tuning {algo_name.upper()}", file=sys.stdout, dynamic_ncols=True)
    def _cb(st, tr):
        pbar.update(1)
        try: 
            pbar.set_postfix(best_val=f"{st.best_value:.1f}")
        except: 
            pbar.set_postfix(best_val="–")
        sys.stdout.write("\n")  # Add newline after Optuna's output
        sys.stdout.flush()
    study.optimize(lambda t: objective(t, algo_name), n_trials=n_trials, n_jobs=n_parallel, callbacks=[_cb])
    pbar.close()

    os.makedirs("jax_hp_results", exist_ok=True)
    with open(f"jax_hp_results/{algo_name}_best_params.json", "w") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value, "n_trials": n_trials}, f, indent=4)

    print(f"Best reward: {study.best_value:.2f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    return study.best_params

if __name__ == "__main__":
    for algo in [ "td3", "ddpg", "sac", "dqn"]:
        tune_hyperparameters(algo)
