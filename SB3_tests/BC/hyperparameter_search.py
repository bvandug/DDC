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

# --- Import Environments ---
from PYBCEnv import BuckConverterEnv as BCPyEnv
from full_wrap.BC.BCSimulinkEnv import BCSimulinkEnv

# --- Constants ---
TOTAL_TIMESTEPS_PER_TRIAL = 400_000
EVAL_TARGET_VOLTAGE = 30.0
N_EVAL_EPISODES = 1
TB_ROOT = "./buck_converter_tuning_logs/"
os.makedirs(TB_ROOT, exist_ok=True)


def define_hyperparameters(trial: optuna.Trial):
    """Defines the extended hyperparameter search space for A2C."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_size = trial.suggest_int("layer_size", 64, 512, log=True)
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

    net_arch = dict(pi=[layer_size] * n_layers, vf=[layer_size] * n_layers)
    policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9995),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.02, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.7),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.2, 2.0),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.995),
        "policy_kwargs": policy_kwargs,
        "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False])
    }
    return params


def objective(trial: optuna.Trial):
    """Objective function: full training then final MATLAB evaluation."""
    temp_model_path = f"temp_model_trial_{trial.number}.zip"
    temp_stats_path = f"temp_stats_trial_{trial.number}.pkl"

    train_env = None
    eval_env = None

    try:
        hyperparams = define_hyperparameters(trial)

        train_env_fn = lambda: BCPyEnv(use_randomized_goal=True)
        train_env = DummyVecEnv([train_env_fn])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        model = A2C("MlpPolicy", train_env, verbose=0, **hyperparams)
        model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_TRIAL, reset_num_timesteps=False, progress_bar=False)

        model.save(temp_model_path)
        train_env.save(temp_stats_path)

        eval_env_fn = lambda: BCSimulinkEnv(
            model_name="bcSim",
            enable_plotting=False,
            target_voltage=EVAL_TARGET_VOLTAGE,
            max_episode_time=0.01
        )
        eval_env = DummyVecEnv([eval_env_fn])
        eval_env = VecNormalize.load(temp_stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        eval_model = A2C.load(temp_model_path, env=eval_env)
        mean_reward, _ = evaluate_policy(eval_model, eval_env, n_eval_episodes=N_EVAL_EPISODES)

        print(f"[RESULT] Trial {trial.number} final mean reward: {mean_reward:.2f}")
        return mean_reward

    except Exception as e:
        print(f"[FAIL] Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()

    finally:
        if train_env: train_env.close()
        if eval_env: eval_env.close()
        if os.path.exists(temp_model_path): os.remove(temp_model_path)
        if os.path.exists(temp_stats_path): os.remove(temp_stats_path)


def tune_hyperparameters(algo_name, n_trials=100, n_jobs=1):
    study_name = f"{algo_name}-pytrain-matlabeval-wide"
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.NopPruner()  # No pruning
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
        print("\nInterrupted by user. You can resume this study later.")
    finally:
        pbar.close()

    results_dir = "hyperparameter_results_matlab_eval"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{algo_name}_best_params.json")

    results = {"best_value": study.best_value, "best_params": study.best_params}
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*80)
    print(f"Best parameters saved to: {results_file}")
    print(f"Best value (mean reward): {study.best_value:.2f}")

    # Sort and print all completed trials from best to worst
    sorted_trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)

    for rank, trial in enumerate(sorted_trials):
        print(f"\n#{rank + 1}: Trial {trial.number} | Reward: {trial.value:.2f}")
        for k, v in trial.params.items():
            print(f"  {k}: {v}")

    return study.best_params


if __name__ == "__main__":
    tune_hyperparameters("A2C", n_trials=150, n_jobs=1)
