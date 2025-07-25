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
from stable_baselines3.common.monitor import Monitor

# --- Import PPO ---
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# --- Import Environments ---
from PYBCEnv import BuckConverterEnv as BCPyEnv
from BCSimulinkEnv import BCSimulinkEnv

# --- Constants ---
TOTAL_TIMESTEPS_PER_TRIAL = 400_000
EVAL_TARGET_VOLTAGE = 30.0
N_EVAL_EPISODES = 1
TB_ROOT = "./buck_converter_tuning_logs/"
os.makedirs(TB_ROOT, exist_ok=True)

# --- ADDED: Constants for Pruning ---
EVAL_INTERVAL = 50_000  # Evaluate every 50k steps for pruning
MIN_RESOURCES_FOR_PRUNING = 100_000 # Start pruning after the 2nd evaluation


def define_hyperparameters(trial: optuna.Trial):
    """
    Defines the hyperparameter search space for PPO.
    """
    # Network architecture parameters
    n_layers = trial.suggest_int("n_layers", 1, 3) 
    layer_size = trial.suggest_int("layer_size", 32, 512)
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu", "elu"])
    activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
    activation_fn = activation_map[activation_fn_name]
    
    net_arch = [layer_size] * n_layers
    policy_kwargs = {
        "net_arch": dict(pi=net_arch, vf=net_arch),
        "activation_fn": activation_fn
    }

    # PPO-specific rollout parameters
    n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    # Prune trials with invalid batch_size to n_steps ratio
    if batch_size > n_steps or n_steps % batch_size != 0:
        raise optuna.TrialPruned()

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 2e-4, 2e-3, log=True),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": trial.suggest_int("n_epochs", 4, 20),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "policy_kwargs": policy_kwargs
    }
    return params


def objective(trial: optuna.Trial):
    """
    Objective function with intermediate Python evaluation for pruning.
    """
    seed = random.randint(0, 1_000_000)
    set_random_seed(seed)
    trial.set_user_attr("seed", seed)
    print(f"\n[INFO] Starting Trial {trial.number} with Seed: {seed}")
    
    temp_model_path = f"temp_model_trial_{trial.number}.zip"
    temp_stats_path = f"temp_stats_trial_{trial.number}.pkl"

    train_env = None
    final_eval_env = None

    try:
        hyperparams = define_hyperparameters(trial)

        # --- UPDATED: Removed GymWrapper ---
        train_env_fn = lambda: Monitor(BCPyEnv(use_randomized_goal=True))
        train_env = DummyVecEnv([train_env_fn])
        train_env.seed(seed)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        model = PPO("MlpPolicy", train_env, verbose=0, tensorboard_log=os.path.join(TB_ROOT, "PPO"), seed=seed, **hyperparams)
        
        # --- TRAINING AND PRUNING LOOP ---
        timesteps_so_far = 0
        while timesteps_so_far < TOTAL_TIMESTEPS_PER_TRIAL:
            model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False, progress_bar=False)
            timesteps_so_far += EVAL_INTERVAL
            
            # --- Intermediate evaluation on the fast Python environment ---
            model.save(temp_model_path)
            train_env.save(temp_stats_path)
            
            # Create a temporary Python env for the intermediate check
            # --- UPDATED: Removed GymWrapper ---
            temp_py_eval_env_fn = lambda: Monitor(BCPyEnv(fixed_goal_voltage=EVAL_TARGET_VOLTAGE))
            temp_py_eval_env = DummyVecEnv([temp_py_eval_env_fn])
            temp_py_eval_env.seed(seed)
            temp_py_eval_env = VecNormalize.load(temp_stats_path, temp_py_eval_env)
            temp_py_eval_env.training = False
            
            intermediate_model = PPO.load(temp_model_path, env=temp_py_eval_env)
            mean_reward, _ = evaluate_policy(intermediate_model, temp_py_eval_env, n_eval_episodes=3)
            temp_py_eval_env.close()
            
            print(f"[DEBUG] Trial {trial.number} | Timestep {timesteps_so_far} | Intermediate Reward: {mean_reward:.2f}")

            trial.report(mean_reward, timesteps_so_far)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # --- FINAL EVALUATION ON MATLAB (only for trials that complete training) ---
        print("\n--- Training complete. Proceeding to final MATLAB evaluation. ---")
        
        # --- UPDATED: Removed GymWrapper ---
        final_eval_env_fn = lambda: BCSimulinkEnv(
            model_name="bcSim",
            enable_plotting=False,
            fixed_goal_voltage=EVAL_TARGET_VOLTAGE,
            max_episode_time=0.01
        )
        final_eval_env = DummyVecEnv([final_eval_env_fn])
        final_eval_env.seed(seed)
        final_eval_env = VecNormalize.load(temp_stats_path, final_eval_env)
        final_eval_env.training = False
        final_eval_env.norm_reward = False

        final_model = PPO.load(temp_model_path, env=final_eval_env)
        final_mean_reward, _ = evaluate_policy(final_model, final_eval_env, n_eval_episodes=N_EVAL_EPISODES)

        print(f"[RESULT] Trial {trial.number} final MATLAB mean reward: {final_mean_reward:.2f}")
        return final_mean_reward

    except optuna.exceptions.TrialPruned as e:
        # Re-raise pruned trials to be handled by Optuna
        raise e
    except Exception as e:
        print(f"[FAIL] Trial {trial.number} failed: {e}")
        # Prune on other exceptions to avoid crashing the study
        raise optuna.TrialPruned()

    finally:
        if train_env: train_env.close()
        if final_eval_env: final_eval_env.close()
        if os.path.exists(temp_model_path): os.remove(temp_model_path)
        if os.path.exists(temp_stats_path): os.remove(temp_stats_path)


def tune_hyperparameters(n_trials=100, n_jobs=1):
    algo_name = "PPO"
    study_name = f"{algo_name}-pytrain-matlabeval-new-final-updated"
    storage_name = f"sqlite:///{study_name}.db"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, 
        n_warmup_steps=MIN_RESOURCES_FOR_PRUNING
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner
    )

    pbar = tqdm(total=n_trials, desc=f"Tuning {algo_name.upper()}", file=sys.stdout)

    def _pbar_callback(study, trial):
        pbar.update(1)
        try:
            pbar.set_postfix(best_val=f"{study.best_value:.2f}")
        except (ValueError, TypeError):
            pbar.set_postfix(best_val="N/A")

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[_pbar_callback])
    except KeyboardInterrupt:
        print("\nInterrupted by user. You can resume this study later.")
    finally:
        pbar.close()

    results_dir = "hyperparameter_results_matlab_eval-final"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{algo_name}_best_params.json")

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

    print("\n" + "="*80)
    print(f"All trial data saved to: {results_file}")
    
    try:
        if not study.best_trial:
            print("No trials completed successfully.")
            return None
    except ValueError:
        print("No trials completed successfully.")
        return None

    print("\n--- INDIVIDUAL TRIAL RESULTS (BEST TO WORST) ---")
    print("="*80)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)

    for rank, trial in enumerate(sorted_trials):
        seed = trial.user_attrs.get('seed', 'N/A')
        print(f"\n--- Rank {rank + 1}: Trial #{trial.number} (Seed: {seed}) ---")
        print(f"  Value (Mean Reward): {trial.value:.2f}")
        print("  Params: ")
        for k, v in trial.params.items():
            print(f"    {k:<20}: {v}")
            
    print("\n" + "="*80)
    print(f"--- Best parameters for {algo_name} ---")
    best_trial = study.best_trial
    best_seed = best_trial.user_attrs.get('seed', 'N/A')
    print(f"Best value (mean reward): {best_trial.value:.2f}")
    print(f"Best trial was #{best_trial.number} with seed {best_seed}")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    return study.best_params


if __name__ == "__main__":
    tune_hyperparameters(n_trials=110, n_jobs=1)
