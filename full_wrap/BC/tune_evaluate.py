import os
import torch
import optuna
import numpy as np
import json
import time
from torch import nn
from tqdm import tqdm
import sys

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# --- Import Both Environments ---
# Python environment for fast training
from PYBCEnv import BuckConverterEnv as BCPyEnv 
# MATLAB/Simulink environment for accurate evaluation
from full_wrap.BC.BCSimulinkEnv import BCSimulinkEnv 

# --- CONSTANTS (Optimized for Colab Runtime) ---
# Total training time for a single trial
TOTAL_TIMESTEPS_PER_TRIAL = 100000 
# How often to stop training and run a MATLAB evaluation
EVAL_INTERVAL = 20000
# Minimum steps before a trial can be pruned
MIN_RESOURCES_FOR_PRUNING = 40000 
# Voltage used for evaluation in the MATLAB environment
EVAL_TARGET_VOLTAGE = 30.0 
# Number of evaluation episodes (1 is sufficient for a deterministic environment)
N_EVAL_EPISODES = 1
# Root directory for TensorBoard logs
TB_ROOT = "./buck_converter_tuning_logs/"
os.makedirs(TB_ROOT, exist_ok=True)

def define_hyperparameters(trial: optuna.Trial):
    """Defines the hyperparameter search space for A2C."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_size = trial.suggest_int("layer_size", 128, 384, log=True)
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]
    
    net_arch = dict(pi=[layer_size] * n_layers, vf=[layer_size] * n_layers)
    policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}
    
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.005, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.4, 0.6),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.4, 1.0),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.92, 0.98),
        "policy_kwargs": policy_kwargs,
        "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False])
    }
    return params

def objective(trial: optuna.Trial):
    """
    The objective function for Optuna. Trains in Python, evaluates in MATLAB.
    """
    temp_model_path = f"temp_model_trial_{trial.number}.zip"
    temp_stats_path = f"temp_stats_trial_{trial.number}.pkl"

    train_env = None
    eval_env = None
    
    try:
        hyperparams = define_hyperparameters(trial)
        
        train_env_fn = lambda: Monitor(BCPyEnv(use_randomized_goal=True))
        train_env = DummyVecEnv([train_env_fn])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        model = A2C("MlpPolicy", train_env, verbose=0, **hyperparams)
        
        timesteps_so_far = 0
        while timesteps_so_far < TOTAL_TIMESTEPS_PER_TRIAL:
            model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False, progress_bar=False)
            timesteps_so_far += EVAL_INTERVAL
            
            model.save(temp_model_path)
            train_env.save(temp_stats_path)
            
            eval_env_fn = lambda: Monitor(BCSimulinkEnv(model_name="bcSim", enable_plotting=False, target_voltage=EVAL_TARGET_VOLTAGE, max_episode_time=0.01))
            eval_env = DummyVecEnv([eval_env_fn])
            eval_env = VecNormalize.load(temp_stats_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            eval_model = A2C.load(temp_model_path, env=eval_env)
            
            mean_reward, _ = evaluate_policy(eval_model, eval_env, n_eval_episodes=N_EVAL_EPISODES)
            
            eval_env.close()
            eval_env = None
            
            trial.report(mean_reward, timesteps_so_far)
            
            if trial.should_prune():
                raise optuna.TrialPruned()

        return mean_reward

    except Exception as e:
        print(f"[FAIL] Trial {trial.number} failed with an exception: {e}")
        raise optuna.TrialPruned()
    finally:
        if train_env: train_env.close()
        if eval_env: eval_env.close()
        if os.path.exists(temp_model_path): os.remove(temp_model_path)
        if os.path.exists(temp_stats_path): os.remove(temp_stats_path)

def tune_hyperparameters(algo_name, n_trials=25, n_jobs=1):
    """Main function to set up and run the Optuna study."""
    study_name = f"{algo_name}-pytrain-matlabeval-colab"
    storage_name = f"sqlite:///{study_name}.db"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=MIN_RESOURCES_FOR_PRUNING // EVAL_INTERVAL)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name,
        load_if_exists=True, direction="maximize", pruner=pruner
    )

    pbar = tqdm(total=n_trials, desc=f"Tuning {algo_name.upper()}", file=sys.stdout)
    def _pbar_callback(study, trial):
        pbar.update(1)
        try:
            pbar.set_postfix(best_val=f"{study.best_value:.2f}")
        except ValueError:
            pbar.set_postfix(best_val="N/A")

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, callbacks=[_pbar_callback])
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current results.")
    finally:
        pbar.close()

    results_dir = "hyperparameter_results_matlab_eval"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{algo_name}_best_params.json")
    
    results = {"best_value": study.best_value, "best_params": study.best_params}
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n\n" + "="*80)
    print(f"--- Best parameters for {algo_name} saved to {results_file} ---")
    print(f"Best value (mean reward): {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study.best_params

if __name__ == "__main__":
    # n_jobs must be 1 because each trial launches a MATLAB instance.
    tune_hyperparameters(
        algo_name="A2C",
        n_trials=25, 
        n_jobs=1 
    )
