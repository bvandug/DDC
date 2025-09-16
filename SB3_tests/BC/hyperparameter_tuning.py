import os
import json
import sys
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn
from tqdm import tqdm

from BCPythonEnv import BuckConverterEnv as BCPyEnv, DiscretizeActionWrapper

# CONSTANTS FOR PRUNING AND EVALUATION
EVAL_INTERVAL = 10000
MIN_RESOURCES_FOR_PRUNING = 40000
TB_ROOT = "./buck_converter_tuning_logs/"
os.makedirs(TB_ROOT, exist_ok=True)


def define_hyperparameters(trial: optuna.Trial, algo_name: str):
    """Defines the hyperparameter search space for a given algorithm.

    This function is called by Optuna for each trial to sample a set of
    hyperparameters from a pre-defined distribution.

    Args:
        trial (optuna.Trial): The Optuna trial object.
        algo_name (str): The name of the algorithm to define HPs for.

    Returns:
        dict: A dictionary of sampled hyperparameters for the given algorithm.
    """
    algo = algo_name.lower()
    activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU,
                      "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}

    # Define Network Architecture
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_size = trial.suggest_int("layer_size", 64, 256, log=True)
    activation_fn_name = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu", "leaky_relu", "elu"]
    )
    activation_fn = activation_map[activation_fn_name]
    net_arch = [layer_size] * n_layers

    # On-Policy Algorithms (A2C, PPO)
    if algo in ["a2c", "ppo"]:
        policy_kwargs = {
            "net_arch": dict(pi=net_arch, vf=net_arch),
            "activation_fn": activation_fn
        }
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2,
                                                 log=True),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.01, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.2, 0.8),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
            "policy_kwargs": policy_kwargs,
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0)
        }
        if algo == "a2c":
            params["n_steps"] = trial.suggest_int("n_steps", 16, 2048,
                                                  log=True)
        elif algo == "ppo":
            n_steps = trial.suggest_categorical(
                "n_steps", [128, 256, 512, 1024, 2048, 4096]
            )
            batch_size = trial.suggest_categorical("batch_size",
                                                   [64, 128, 256, 512])
            params["n_steps"] = n_steps
            params["batch_size"] = min(batch_size, n_steps)
            params["n_epochs"] = trial.suggest_int("n_epochs", 4, 20)
            params["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4)
        return params

    # Off-Policy Algorithms (SAC, TD3, DDPG, DQN)
    else:
        if algo == "sac":
            policy_kwargs = {"net_arch": dict(pi=net_arch, qf=net_arch),
                             "activation_fn": activation_fn}
        else:
            policy_kwargs = {"net_arch": net_arch,
                             "activation_fn": activation_fn}

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3,
                                                 log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
            "batch_size": trial.suggest_categorical("batch_size",
                                                    [64, 128, 256, 512]),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "policy_kwargs": policy_kwargs
        }

        if algo in ["sac", "td3", "ddpg"]:
            params["tau"] = trial.suggest_float("tau", 0.001, 0.02)
            noise_sigma = trial.suggest_float("action_noise_sigma", 0.1, 0.5)
            params["action_noise"] = NormalActionNoise(
                mean=np.zeros(1), sigma=noise_sigma * np.ones(1)
            )

        if algo == "td3":
            params["policy_delay"] = trial.suggest_int("policy_delay", 1, 4)
            params["target_policy_noise"] = trial.suggest_float(
                "target_policy_noise", 0.1, 0.5
            )
            params["target_noise_clip"] = trial.suggest_float(
                "target_noise_clip", 0.3, 0.7
            )

        if algo == "dqn":
            params["learning_starts"] = trial.suggest_int("learning_starts",
                                                          5000, 20000)
            params["exploration_fraction"] = trial.suggest_float(
                "exploration_fraction", 0.05, 0.5
            )
            params["exploration_final_eps"] = trial.suggest_float(
                "exploration_final_eps", 0.01, 0.2
            )
            params["train_freq"] = trial.suggest_categorical("train_freq",
                                                             [1, 4, 8, 16, 32])
            params["target_update_interval"] = trial.suggest_int(
                "target_update_interval", 500, 2000
            )

        return params


def objective(trial: optuna.Trial, algo_name: str, total_timesteps: int):
    """The objective function for Optuna to minimize/maximize.

    This function defines a single trial: it creates a model with a sampled
    set of hyperparameters, trains it in intervals, evaluates its performance,
    and reports the results back to Optuna for pruning or ranking.

    Args:
        trial (optuna.Trial): The Optuna trial object for this run.
        algo_name (str): The name of the algorithm being tuned.
        total_timesteps (int): The total number of timesteps to train for.

    Returns:
        float: The mean reward achieved during the final evaluation, which
               Optuna will use as the objective value to maximize.
    """
    seed = 42
    set_random_seed(seed)
    trial.set_user_attr("seed", seed)

    print(f"\n[INFO] Starting Trial {trial.number} for {algo_name.upper()} "
          f"({total_timesteps} steps)")

    def make_train_env():
        env = BCPyEnv(use_randomized_goal=True)
        if algo_name.lower() == "dqn":
            env = DiscretizeActionWrapper(env, n_bins=17)
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])
    train_env.seed(seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False,
                             clip_obs=10.0)

    hyperparams = define_hyperparameters(trial, algo_name)
    hyperparams['seed'] = seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if algo_name.lower() in ["a2c", "ppo"]:
        device = "cpu"  # On-policy algos often faster on CPU for simple envs

    if algo_name.lower() in ["sac", "td3", "ddpg", "dqn"] and \
       "learning_starts" not in hyperparams:
        hyperparams["learning_starts"] = 10000

    model_class = globals()[algo_name.upper()]
    model = model_class(
        "MlpPolicy", train_env, device=device, verbose=0,
        tensorboard_log=os.path.join(TB_ROOT, algo_name), **hyperparams
    )

    timesteps_so_far = 0
    mean_reward = -np.inf

    try:
        while timesteps_so_far < total_timesteps:
            model.learn(total_timesteps=EVAL_INTERVAL,
                        reset_num_timesteps=False,
                        tb_log_name=f"trial_{trial.number}")
            timesteps_so_far += EVAL_INTERVAL

            # Intermediate Evaluation for Pruning
            def make_eval_env():
                eval_env = BCPyEnv(use_randomized_goal=False,
                                   fixed_goal_voltage=30.0)
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
        return -1e9  # Return a very low value for failed trials
    finally:
        train_env.close()
        if os.path.exists("temp_vec_normalize.pkl"):
            os.remove("temp_vec_normalize.pkl")


def tune_hyperparameters(algo_name, n_trials=50, n_jobs=1,
                         total_timesteps=100000):
    """Main function to set up and run the Optuna study for an algorithm.

    Args:
        algo_name (str): The algorithm to tune (e.g., 'A2C', 'PPO').
        n_trials (int): The number of trials to run.
        n_jobs (int): The number of parallel jobs to run.
        total_timesteps (int): The total training timesteps for each trial.
    """
    study_name = f"{algo_name}-bc-tuning-seed42"
    storage_name = f"sqlite:///{study_name}.db"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5,
                                         n_warmup_steps=MIN_RESOURCES_FOR_PRUNING)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name,
        load_if_exists=True, direction="maximize", pruner=pruner
    )

    pbar = tqdm(total=n_trials, desc=f"Tuning {algo_name.upper()}",
                file=sys.stdout)

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

    # Save Ranked Results
