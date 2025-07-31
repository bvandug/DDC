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

from stable_baselines3 import PPO, SAC, A2C, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Assumes your environment file is named 'PYBCEnv.py' and is accessible
from PYBCEnv import BuckConverterEnv as BCPyEnv

# --- CONSTANTS ---
TOTAL_TIMESTEPS_PER_TRIAL = 100000
EVAL_INTERVAL = 10000
MIN_RESOURCES_FOR_PRUNING = 20000
TB_ROOT = "./buck_converter_tuning_logs/"
os.makedirs(TB_ROOT, exist_ok=True)

def define_hyperparameters(trial: optuna.Trial, algo_name: str):
    """Defines the CORE hyperparameter search space for a given algorithm."""
    algo = algo_name.lower()
    
    if algo in ["a2c", "ppo"]:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_size = trial.suggest_int("layer_size", 64, 512, log=True)
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]
        
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
        
        # --- CORRECTED LOGIC FOR TRAIN_FREQ AND GRADIENT_STEPS ---
        # 1. Sample both from their full, static list of choices
        gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4, 8, 16, 32, 64, 128])
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256])

        # 2. After sampling, enforce the constraint by correcting invalid combinations
        if train_freq < gradient_steps:
            # If the combination is invalid, adjust train_freq to a valid value.
            train_freq = gradient_steps

        params["gradient_steps"] = gradient_steps
        params["train_freq"] = train_freq
        
        return params

    elif algo == "ddpg":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layer_size = trial.suggest_int("layer_size", 64, 512, log=True)
        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]
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

    train_env_fn = lambda: Monitor(BCPyEnv(use_randomized_goal=True))
    train_env = DummyVecEnv([train_env_fn])
    train_env.seed(seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    hyperparams = define_hyperparameters(trial, algo_name)
    hyperparams['seed'] = seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if algo_name.lower() in ["sac", "td3", "ddpg"]:
        hyperparams["learning_starts"] = 10000

    model_class = globals()[algo_name.upper()]
    model = model_class("MlpPolicy", train_env, device=device, verbose=0, 
                        tensorboard_log=os.path.join(TB_ROOT, algo_name), **hyperparams)

    timesteps_so_far = 0
    mean_reward = -np.inf
    
    try:
        while timesteps_so_far < TOTAL_TIMESTEPS_PER_TRIAL:
            model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False,
                        tb_log_name=f"trial_{trial.number}")
            timesteps_so_far += EVAL_INTERVAL

            eval_env_fn = lambda: Monitor(BCPyEnv(use_randomized_goal=False, fixed_goal_voltage=30.0))
            eval_env = DummyVecEnv([eval_env_fn])
            eval_env.seed(seed)
            train_env.save("temp_vec_normalize.pkl")
            eval_env = VecNormalize.load("temp_vec_normalize.pkl", eval_env)
            eval_env.training = False 
            eval_env.norm_reward = False
            
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1)
            eval_env.close()
            
            print(f"[DEBUG] Trial {trial.number} | Timestep {timesteps_so_far} | Mean reward: {mean_reward}")
            
            trial.report(mean_reward, timesteps_so_far)
            
            if trial.should_prune():
                print(f"[PRUNE] Trial {trial.number} pruned at {timesteps_so_far} steps.")
                raise optuna.TrialPruned()

        return mean_reward
        
    except (AssertionError, ValueError) as e:
        print(f"[FAIL] Trial {trial.number} failed with error: {e}")
        # Returning a value like -inf or a very low number can be better than pruning
        # as it still provides information to some samplers.
        return -1e9 # Return a very bad value instead of pruning
    finally:
        train_env.close()
        if os.path.exists("temp_vec_normalize.pkl"):
            os.remove("temp_vec_normalize.pkl")

def tune_hyperparameters(algo_name, n_trials=50, n_jobs=1):
    """
    Main function to set up and run the Optuna study.
    """
    study_name = f"{algo_name}-bc-tuning-final-seeded-v4"
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
    ALGORITHM_TO_TUNE = "TD3"
    tune_hyperparameters(
        algo_name=ALGORITHM_TO_TUNE,
        n_trials=50, 
        n_jobs=1 
    )