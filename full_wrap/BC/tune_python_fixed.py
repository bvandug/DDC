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
import gymnasium as gym

# --- Import All Algorithms ---
from stable_baselines3 import PPO, SAC, A2C, TD3, DDPG, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Assumes your environment file is named 'PYBCEnv.py' and is accessible
from PYBCEnv import BuckConverterEnv as BCPyEnv

# --- CONSTANTS ---
TOTAL_TIMESTEPS_PER_TRIAL = 300000
EVAL_INTERVAL = 20000
MIN_RESOURCES_FOR_PRUNING = 60000
TB_ROOT = "./buck_converter_tuning_logs/"
os.makedirs(TB_ROOT, exist_ok=True)

# --- Wrapper to make continuous environment compatible with DQN ---
class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    A wrapper to discretize a continuous action space for DQN.
    :param env: The continuous action environment to wrap.
    :param n_bins: The number of discrete actions to create.
    """
    def __init__(self, env, n_bins=17):
        super().__init__(env)
        self.n_bins = n_bins
        self.action_space = gym.spaces.Discrete(self.n_bins)
        self.continuous_actions = np.linspace(
            self.env.action_space.low[0],
            self.env.action_space.high[0],
            self.n_bins
        )

    def action(self, action):
        """
        Translates the discrete action from the agent into its
        corresponding continuous value for the environment.
        """
        continuous_action = self.continuous_actions[action]
        return np.array([continuous_action], dtype=np.float32)

def define_hyperparameters(trial: optuna.Trial, algo_name: str):
    """Defines the CORE hyperparameter search space for a given algorithm."""
    algo = algo_name.lower()
    
    if algo == "ppo":
        n_layers = trial.suggest_int("n_layers", 1, 5)
        layer_size = trial.suggest_int("layer_size", 64, 512, log=True)
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]
        
        net_arch = dict(pi=[layer_size] * n_layers, vf=[layer_size] * n_layers)
        policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}
        
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        if batch_size > n_steps: batch_size = n_steps
            
        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": trial.suggest_int("n_epochs", 4, 20),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.01, log=True),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "vf_coef": trial.suggest_float("vf_coef", 0.2, 0.8),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
            "policy_kwargs": policy_kwargs
        }

    elif algo == "sac":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_size = trial.suggest_int("layer_size", 32, 256, log=True) 
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu", "elu"])
        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
        activation_fn = activation_map[activation_fn_name]
        net_arch = [layer_size] * n_layers
        policy_kwargs = {"net_arch": dict(pi=net_arch, qf=net_arch), "activation_fn": activation_fn}

        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.001, 0.01, 0.1]),
            "policy_kwargs": policy_kwargs
        }

    # --- UPDATED: DQN Hyperparameters ---
    elif algo == "dqn":
        n_layers = trial.suggest_int("n_layers", 1, 5)
        layer_size = trial.suggest_int("layer_size", 64, 256, log=True)
        # Fixed typo and expanded options
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu", "elu"])
        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
        activation_fn = activation_map[activation_fn_name]
        
        policy_kwargs = {"net_arch": [layer_size] * n_layers, "activation_fn": activation_fn}
        
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "tau": trial.suggest_float("tau", 0.005, 0.02),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
            #"gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4, 8]),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
            "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
            # Added target_update_interval
            "target_update_interval": trial.suggest_categorical("target_update_interval", [500, 1000, 2000, 5000, 10000]),
            "policy_kwargs": policy_kwargs
        }
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")

def objective(trial: optuna.Trial, algo_name: str):
    """
    The objective function for Optuna to minimize/maximize.
    """
    seed = random.randint(0, 1_000_000)
    set_random_seed(seed)
    trial.set_user_attr("seed", seed)
    print(f"\n[INFO] Starting Trial {trial.number} with Seed: {seed}")

    # Conditionally wrap environment for DQN
    if algo_name.lower() == 'dqn':
        # 9 bins for [1, 2, ..., 9]. 17 bins for [1.0, 1.5, ..., 9.0]
        train_env_fn = lambda: Monitor(DiscretizeActionWrapper(BCPyEnv(use_randomized_goal=True), n_bins=17))
    else:
        train_env_fn = lambda: Monitor(BCPyEnv(use_randomized_goal=True))

    train_env = DummyVecEnv([train_env_fn])
    train_env.seed(seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    hyperparams = define_hyperparameters(trial, algo_name)
    hyperparams['seed'] = seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if algo_name.lower() in ["sac", "td3", "ddpg", "dqn"]:
        hyperparams["learning_starts"] = 10000

    model_class = globals()[algo_name.upper()]
    model = model_class("MlpPolicy", train_env, device=device, verbose=0, 
                        tensorboard_log=os.path.join(TB_ROOT, algo_name), **hyperparams)

    timesteps_so_far = 0
    
    try:
        while timesteps_so_far < TOTAL_TIMESTEPS_PER_TRIAL:
            model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False,
                        tb_log_name=f"trial_{trial.number}")
            timesteps_so_far += EVAL_INTERVAL

            # Conditionally wrap evaluation environment for DQN
            if algo_name.lower() == 'dqn':
                eval_env_fn = lambda: Monitor(DiscretizeActionWrapper(BCPyEnv(use_randomized_goal=False, fixed_goal_voltage=30.0), n_bins=17))
            else:
                eval_env_fn = lambda: Monitor(BCPyEnv(use_randomized_goal=False, fixed_goal_voltage=30.0))
            
            eval_env = DummyVecEnv([eval_env_fn])
            eval_env.seed(seed)
            train_env.save("temp_vec_normalize.pkl")
            eval_env = VecNormalize.load("temp_vec_normalize.pkl", eval_env)
            eval_env.training = False 
            eval_env.norm_reward = False
            
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=3)
            eval_env.close()
            
            print(f"[DEBUG] Trial {trial.number} | Timestep {timesteps_so_far} | Mean reward: {mean_reward}")
            
            trial.report(mean_reward, timesteps_so_far)
            
            if trial.should_prune():
                print(f"[PRUNE] Trial {trial.number} pruned at {timesteps_so_far} steps.")
                raise optuna.TrialPruned()

        return mean_reward
        
    except (AssertionError, ValueError) as e:
        print(f"[FAIL] Trial {trial.number} failed with error: {e}")
        return -1e9
    finally:
        train_env.close()
        if os.path.exists("temp_vec_normalize.pkl"):
            os.remove("temp_vec_normalize.pkl")

def tune_hyperparameters(algo_name, n_trials=50, n_jobs=1):
    """
    Main function to set up and run the Optuna study.
    """
    study_name = f"{algo_name}-bc-tuning-modular-v1-17bins"
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
            lambda trial: objective(trial, algo_name),
            n_trials=n_trials, n_jobs=n_jobs,
            callbacks=[_pbar_callback]
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current results.")
    finally:
        pbar.close()

    # --- UPDATED: Full saving and printing logic ---
    results_dir = "hyperparameter_results_final"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{algo_name}_best_params_seeded.json")

    all_trials_data = []
    for trial in study.trials:
        trial_data = {
            "number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
            "params": trial.params,
            "user_attrs": trial.user_attrs
        }
        all_trials_data.append(trial_data)

    with open(results_file, "w") as f:
        json.dump(all_trials_data, f, indent=4)
        
    print("\n\n" + "="*80)
    print(f"All trial data saved to: {results_file}")
    
    # --- Print a ranked list of all completed trials ---
    print("--- INDIVIDUAL TRIAL RESULTS (BEST TO WORST) ---")
    print("="*80)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    
    if completed_trials:
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        for i, trial in enumerate(sorted_trials):
            seed = trial.user_attrs.get('seed', 'N/A')
            print(f"\n--- Rank {i+1}: Trial #{trial.number} (Seed: {seed}) ---")
            print(f"  Value (Mean Reward): {trial.value:.4f}")
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key:<20}: {value}")
    else:
        print("No trials completed successfully.")
            
    print("\n" + "="*80)
    print(f"--- Best parameters for {algo_name} ---")
    
    try:
        best_trial = study.best_trial
        print(f"Best value (mean reward): {best_trial.value:.4f}")
        best_seed = best_trial.user_attrs.get('seed', 'N/A')
        print(f"Best trial was #{best_trial.number} with seed {best_seed}")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        return study.best_params
    except (ValueError, AttributeError):
         print("No best trial found.")
         return None


if __name__ == "__main__":
    ALGORITHM_TO_TUNE = "DQN"
    tune_hyperparameters(
        algo_name=ALGORITHM_TO_TUNE,
        n_trials=100, 
        n_jobs=1 
    )
