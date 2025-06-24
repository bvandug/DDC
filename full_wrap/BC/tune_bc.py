import optuna
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from BCSimTestEnv import BCSimulinkEnv  # <-- 1. IMPORT YOUR BC ENVIRONMENT
import json
import os
import torch
from torch import nn
from tqdm import tqdm
import sys

# This is the objective function that Optuna will try to maximize.
# It takes a 'trial' object, which it uses to sample hyperparameters.
def objective(trial, algo_name):
    
    # --- Environment Setup ---
    # We create a function to instantiate the environment.
    # We disable plotting and any extra features for speed during tuning.
    env_fn = lambda: BCSimulinkEnv(
        model_name="bcSim",         # <-- 2. USE YOUR BC MODEL
        enable_plotting=False,
        max_episode_time=0.01       # Use a shorter time for faster tuning trials
    )
    
    # Wrap the environment in DummyVecEnv and VecNormalize, just like in training/testing
    env = DummyVecEnv([env_fn])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Hyperparameter Sampling ---
        # The structure is the same as your pendulum script, but adapted for the BC problem.
        # We can reuse most of the parameter ranges as they are generally good starting points.
        
        if algo_name.lower() == "td3":
            # Suggest hyperparameters for the TD3 algorithm
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size": trial.suggest_int("buffer_size", 20000, 100000),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                "tau": trial.suggest_float("tau", 0.005, 0.02),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "action_noise_sigma": trial.suggest_float("action_noise_sigma", 0.05, 0.2),
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "layer_size": trial.suggest_int("layer_size", 64, 256),
                "activation_fn": trial.suggest_categorical("activation_fn", ["tanh", "relu"]),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[params["activation_fn"]]
            policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}

            model = TD3(
                "MlpPolicy", env, verbose=0, device=device,
                learning_rate=params["learning_rate"],
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                gamma=params["gamma"],
                action_noise=NormalActionNoise(mean=np.zeros(1), sigma=params["action_noise_sigma"] * np.ones(1)),
                policy_kwargs=policy_kwargs,
            )

        elif algo_name.lower() == "a2c":
            # Suggest hyperparameters for the A2C algorithm
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.90, 0.999),
                "n_steps": trial.suggest_int("n_steps", 8, 2048, log=True),
                "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.01, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.2, 0.8),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "layer_size": trial.suggest_int("layer_size", 64, 256),
                "activation_fn": trial.suggest_categorical("activation_fn", ["tanh", "relu"]),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[params["activation_fn"]]
            policy_kwargs = {"net_arch": net_arch, "activation_fn": activation_fn}

            model = A2C(
                "MlpPolicy", env, verbose=0, device=device,
                learning_rate=params["learning_rate"],
                gamma=params["gamma"],
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                policy_kwargs=policy_kwargs,
            )

        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        # --- Train and Evaluate ---
        # Train the model for a fixed number of steps
        model.learn(total_timesteps=50000, progress_bar=False) # Reduced for faster trials

        # Evaluate the trained model and calculate the mean reward
        mean_reward = 0
        n_eval_episodes = 5 # Evaluate over a few episodes
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            mean_reward += episode_reward

        # Optuna will try to maximize this return value
        return mean_reward / n_eval_episodes

    finally:
        # Crucially, close the environment to shut down the MATLAB engine
        env.close()


def tune_hyperparameters(algo_name, n_trials=50):
    study_name = f"{algo_name}-bc-tuning"
    storage_name = f"sqlite:///{study_name}.db"
    
    print(f"\nTuning {algo_name.upper()}. Study: {study_name}")
    print(f"Results will be saved to: {storage_name}")

    # Create a study object. The 'storage' argument allows resuming if interrupted.
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize"
    )

    # The main optimization loop
    study.optimize(
        lambda trial: objective(trial, algo_name),
        n_trials=n_trials,
        n_jobs=1,  # IMPORTANT: Set n_jobs to 1 because each trial starts a MATLAB engine
    )

    # --- Save and Print Results ---
    best_params = study.best_params
    best_value = study.best_value
    
    results = {
        "best_value": best_value,
        "best_params": best_params,
    }

    # Save the results to a JSON file
    os.makedirs("hyperparameter_results", exist_ok=True)
    with open(f"hyperparameter_results/{algo_name}_best_params.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n--- Best parameters for {algo_name} ---")
    print(f"Best value (mean reward): {best_value:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params


if __name__ == "__main__":
    # Select the algorithm to tune
    ALGORITHM_TO_TUNE = "a2c" # Options: "a2c", "td3"
    
    print(f"{'=' * 50}")
    print(f"Tuning hyperparameters for {ALGORITHM_TO_TUNE.upper()}...")
    print(f"{'=' * 50}\n")
    
    tune_hyperparameters(
        algo_name=ALGORITHM_TO_TUNE,
        n_trials=50 # Number of different hyperparameter combinations to try
    )