import os
import torch
import optuna
import numpy as np
import json
import time
from torch import nn
from tqdm import tqdm
import sys
import random
from stable_baselines3.common.utils import set_random_seed
import gym
from gym import spaces
from stable_baselines3 import PPO, SAC, A2C, TD3, DDPG, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from PYBCEnv import BuckConverterEnv as BCPyEnv

# --- CONSTANTS ---
EVAL_INTERVAL = 10000
MIN_RESOURCES_FOR_PRUNING = 40000
TB_ROOT = "./buck_converter_tuning_logs/"
os.makedirs(TB_ROOT, exist_ok=True)

class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    A wrapper to discretize a continuous action space for DQN.
    """
    def __init__(self, env, n_bins=17):
        super().__init__(env)
        self.n_bins = n_bins
        # The action space is now an integer from 0 to n_bins-1
        self.action_space = spaces.Discrete(self.n_bins)
        # Create the mapping from the integer action to a continuous value
        self.continuous_actions = np.linspace(
            self.env.action_space.low[0],
            self.env.action_space.high[0],
            self.n_bins
        )

    def action(self, action):
        """
        Translates the discrete action from the agent into its
        corresponding continuous value for the underlying environment.
        """
        # Select the continuous value from the map
        continuous_action = self.continuous_actions[action]
        # Return it in the shape the environment expects
        return np.array([continuous_action], dtype=np.float32)

def define_hyperparameters(trial: optuna.Trial, algo_name: str):
    """Defines the CORE hyperparameter search space for a given algorithm."""
    algo = algo_name.lower()

    if algo in ["a2c", "ppo"]:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_size = trial.suggest_int("layer_size", 64, 256, log=True)
        # UPDATED: Added ELU and LeakyReLU
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu", "elu"])
        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
        activation_fn = activation_map[activation_fn_name]

        net_arch = dict(pi=[layer_size] * n_layers, vf=[layer_size] * n_layers)
        policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.01, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.2, 0.8),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
            "policy_kwargs": policy_kwargs
        }
        if algo == "a2c":
            params["n_steps"] = trial.suggest_int("n_steps", 16, 2048, log=True)
            params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 1.0)
        elif algo == "ppo":
            params["n_steps"] = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048, 4096])
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
            if batch_size > params["n_steps"]:
                batch_size = params["n_steps"]
            params["batch_size"] = batch_size
            params["n_epochs"] = trial.suggest_int("n_epochs", 4, 20)
            params["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4)
            params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 1.0)
        return params

    elif algo == "sac":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_size = trial.suggest_int("layer_size", 32, 256, log=True)
        # This already includes ELU and LeakyReLU
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu", "elu"])
        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
        activation_fn = activation_map[activation_fn_name]
        net_arch = [layer_size] * n_layers
        policy_kwargs = {"net_arch": dict(pi=net_arch, qf=net_arch), "activation_fn": activation_fn}

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.001, 0.01, 0.1]),
            "policy_kwargs": policy_kwargs
        }
        return params

    elif algo == "td3":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_size = trial.suggest_int("layer_size", 32, 256)
        # This already includes ELU and LeakyReLU
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu", "elu"])
        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
        activation_fn = activation_map[activation_fn_name]
        net_arch = [layer_size] * n_layers
        policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}
        action_noise_sigma = trial.suggest_float("action_noise_sigma", 0.1, 0.5)

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
            "batch_size": trial.suggest_int("batch_size", 64, 512),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "policy_delay": trial.suggest_int("policy_delay", 1, 4),
            "action_noise": NormalActionNoise(mean=np.zeros(1), sigma=action_noise_sigma * np.ones(1)),
            "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.5),
            "target_noise_clip": trial.suggest_float("target_noise_clip", 0.3, 0.7),
            "policy_kwargs": policy_kwargs,
        }

        gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4, 8, 16, 32, 64, 128])
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256])
        if train_freq < gradient_steps:
            train_freq = gradient_steps

        params["gradient_steps"] = gradient_steps
        params["train_freq"] = train_freq

        return params

    elif algo == "ddpg":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_size = trial.suggest_int("layer_size", 64, 512, log=True)
        # UPDATED: Added ELU and LeakyReLU
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu", "elu"])
        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
        activation_fn = activation_map[activation_fn_name]
        net_arch = [layer_size] * n_layers
        policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}
        action_noise_sigma = trial.suggest_float("action_noise_sigma", 0.01, 0.2)

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 250_000),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256]),
            "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4, 8, 16, 32, 64, 128]),
            "action_noise": NormalActionNoise(mean=np.zeros(1), sigma=action_noise_sigma * np.ones(1)),
            "policy_kwargs": policy_kwargs
        }
        return params

    elif algo == "dqn":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_size = trial.suggest_int("layer_size", 64, 512, log=True)
        # UPDATED: Added ELU and LeakyReLU
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu", "elu"])
        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
        activation_fn = activation_map[activation_fn_name]
        net_arch = [layer_size] * n_layers
        policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
            "learning_starts": trial.suggest_int("learning_starts", 5000, 20000),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.5),
            "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.2),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32]),
            "target_update_interval": trial.suggest_int("target_update_interval", 500, 2000),
            "policy_kwargs": policy_kwargs
        }
        return params
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

def objective(trial: optuna.Trial, algo_name: str, total_timesteps: int):
    """
    The objective function for Optuna to minimize/maximize.
    """
    seed = 42
    set_random_seed(seed)
    trial.set_user_attr("seed", seed)

    print(f"\n[INFO] Starting Trial {trial.number} for {algo_name.upper()} with Seed: {seed} ({total_timesteps} steps)")

    def make_train_env():
        env = BCPyEnv(use_randomized_goal=True)
        if algo_name.lower() == "dqn":
            env = DiscretizeActionWrapper(env, n_bins=17) 
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])
    train_env.seed(seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    hyperparams = define_hyperparameters(trial, algo_name)
    hyperparams['seed'] = seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if algo_name.lower() in ["a2c", "ppo"]:
      device = "cpu"

    if algo_name.lower() in ["sac", "td3", "ddpg", "dqn"] and "learning_starts" not in hyperparams:
        hyperparams["learning_starts"] = 10000

    model_class = globals()[algo_name.upper()]
    model = model_class("MlpPolicy", train_env, device=device, verbose=0,
                        tensorboard_log=os.path.join(TB_ROOT, algo_name), **hyperparams)

    timesteps_so_far = 0
    mean_reward = -np.inf

    try:
        while timesteps_so_far < total_timesteps:
            model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False,
                        tb_log_name=f"trial_{trial.number}")
            timesteps_so_far += EVAL_INTERVAL

            def make_eval_env():
                eval_env = BCPyEnv(use_randomized_goal=False, fixed_goal_voltage=30.0)
                if algo_name.lower() == "dqn":
                    eval_env = DiscretizeActionWrapper(eval_env, n_bins=17)
                return Monitor(eval_env)
            
            eval_env = DummyVecEnv([make_eval_env])
            eval_env.seed(seed)
            train_env.save("temp_vec_normalize.pkl")
            eval_env = VecNormalize.load("temp_vec_normalize.pkl", eval_env)
            eval_env.training = False
            eval_env.norm_reward = False

            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1)
            eval_env.close()

            trial.report(mean_reward, timesteps_so_far)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return mean_reward

    except (AssertionError, ValueError) as e:
        print(f"[FAIL] Trial {trial.number} failed with error: {e}")
        return -1e9 # Return a very low value for failed trials
    finally:
        train_env.close()
        if os.path.exists("temp_vec_normalize.pkl"):
            os.remove("temp_vec_normalize.pkl")

def tune_hyperparameters(algo_name, n_trials=50, n_jobs=1, total_timesteps=100000):
    """
    Main function to set up and run the Optuna study for a single algorithm.
    """
    study_name = f"{algo_name}-bc-tuning-seed42"
    storage_name = f"sqlite:///{study_name}.db"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=MIN_RESOURCES_FOR_PRUNING)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name,
        load_if_exists=True, direction="maximize", pruner=pruner
    )

    pbar = tqdm(total=n_trials, desc=f"Tuning {algo_name.upper()}", file=sys.stdout)
    def _pbar_callback(study, trial):
        pbar.update(1)
        try:
            pbar.set_postfix(best_val=f"{study.best_value:.2f}")
        except (ValueError, TypeError):
            pbar.set_postfix(best_val="N/A")

    try:
        study.optimize(
            lambda trial: objective(trial, algo_name, total_timesteps),
            n_trials=n_trials, n_jobs=n_jobs,
            callbacks=[_pbar_callback]
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current results.")
    finally:
        pbar.close()

    results_dir = "hyperparameters_SSH"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{algo_name}_all_trials_ranked.json")

    completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    completed_trials.sort(key=lambda t: t.value, reverse=True)

    ranked_trials_data = [
        {"rank": i + 1, "trial_number": t.number, "value": t.value, "params": t.params}
        for i, t in enumerate(completed_trials)
    ]
    
    with open(results_file, "w") as f:
        json.dump(ranked_trials_data, f, indent=4)
    
    print(f"\n--- All ranked trial results saved to: {results_file} ---")

    print("\n\n" + "="*80)
    print(f"--- BEST PARAMETERS FOR {algo_name.upper()} (Seed 42) ---")
    print("="*80)

    try:
        best_trial = study.best_trial
        print(f"Best value (mean reward): {best_trial.value:.4f}")
        print(f"Best trial was #{best_trial.number}")
        print("Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    except (ValueError, AttributeError):
         print("No best trial found for this run.")

if __name__ == "__main__":
    ALGORITHMS_TO_TUNE = ["A2C", "PPO", "SAC", "TD3", "DDPG", "DQN"]

    for algorithm in ALGORITHMS_TO_TUNE:
        print("\n" + "#"*80)
        print(f"# STARTING HYPERPARAMETER TUNING FOR: {algorithm}")
        
        if algorithm in ["A2C", "PPO"]:
            timesteps_for_trial = 200000
        else:
            timesteps_for_trial = 100000
        
        print(f"# Total timesteps per trial: {timesteps_for_trial}")
        print("#"*80 + "\n")

        tune_hyperparameters(
            algo_name=algorithm,
            n_trials=75,
            n_jobs=1,
            total_timesteps=timesteps_for_trial
        )

        print(f"\n--- Finished tuning for {algorithm} ---")

    print("\n\n" + "#"*80)
    print("# ALL HYPERPARAMETER TUNING COMPLETE")
    print("#"*80)