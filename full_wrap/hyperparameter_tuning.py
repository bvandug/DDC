import optuna
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from simulink_env import SimulinkEnv
import json
import os
from tqdm import tqdm
import time

def objective(trial, algo_name, env):
    # Define hyperparameter search space
    if algo_name == "td3":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50000, 200000),
            "batch_size": trial.suggest_int("batch_size", 64, 512),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "policy_delay": trial.suggest_int("policy_delay", 1, 4),
            "action_noise_sigma": trial.suggest_float("action_noise_sigma", 0.1, 0.5),
            "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.5),
            "target_noise_clip": trial.suggest_float("target_noise_clip", 0.3, 0.7),
        }
        
        model = TD3(
            "MlpPolicy", env,
            verbose=0,
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            gamma=params["gamma"],
            train_freq=(1, "step"),
            policy_delay=params["policy_delay"],
            action_noise=NormalActionNoise(mean=np.zeros(1), sigma=params["action_noise_sigma"] * np.ones(1)),
            target_policy_noise=params["target_policy_noise"],
            target_noise_clip=params["target_noise_clip"]
        )
    
    elif algo_name == "sac":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 50000, 200000),
            "batch_size": trial.suggest_int("batch_size", 64, 512),
            "tau": trial.suggest_float("tau", 0.001, 0.02),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "ent_coef": trial.suggest_float("ent_coef", 0.01, 1.0, log=True),
        }
        
        model = SAC(
            "MlpPolicy", env,
            verbose=0,
            learning_rate=params["learning_rate"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            tau=params["tau"],
            gamma=params["gamma"],
            ent_coef=params["ent_coef"]
        )
    
    elif algo_name == "ppo":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_int("n_steps", 64, 2048),
            "batch_size": trial.suggest_int("batch_size", 64, 256),
            "n_epochs": trial.suggest_int("n_epochs", 3, 20),
            "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "ent_coef": trial.suggest_float("ent_coef", 0.01, 0.1, log=True),
        }
        
        model = PPO(
            "MlpPolicy", env,
            verbose=0,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=params["gamma"],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"]
        )

    # Train the model with progress bar
    model.learn(total_timesteps=1000, progress_bar=True)
    
    # Evaluate the model
    mean_reward = 0
    n_eval_episodes = 5
    
    for _ in tqdm(range(n_eval_episodes), desc="Evaluating episodes", leave=False):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        mean_reward += episode_reward
    
    mean_reward /= n_eval_episodes
    
    return mean_reward

def tune_hyperparameters(algo_name, n_trials=50):
    # Initialize the Simulink environment
    env = SimulinkEnv(model_name="PendCart", agent_block="PendCart/RL Agent", dt=0.01)
    
    # Create a study object
    study = optuna.create_study(direction="maximize")
    
    # Create a progress bar for trials
    pbar = tqdm(total=n_trials, desc=f"Tuning {algo_name.upper()}")
    
    # Define callback to update progress bar
    def update_progress(study, trial):
        pbar.update(1)
        pbar.set_postfix({"best_value": f"{study.best_value:.2f}"})
    
    # Run the optimization with progress bar
    study.optimize(
        lambda trial: objective(trial, algo_name, env),
        n_trials=n_trials,
        callbacks=[update_progress]
    )
    
    pbar.close()
    
    # Save the best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    results = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": n_trials
    }
    
    # Create a directory for results if it doesn't exist
    os.makedirs("hyperparameter_results", exist_ok=True)
    
    # Save results to a JSON file
    with open(f"hyperparameter_results/{algo_name}_best_params.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nBest parameters for {algo_name}:")
    print(f"Best value: {best_value}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    env.close()
    return best_params

if __name__ == "__main__":
    # You can tune each algorithm separately
    algorithms = ["td3", "sac", "ppo"]
    
    print("Starting hyperparameter tuning...")
    print("This process will tune each algorithm sequentially:")
    for algo in algorithms:
        print(f"- {algo.upper()}")
    
    for algo in algorithms:
        print(f"\n{'='*50}")
        print(f"Tuning hyperparameters for {algo.upper()}...")
        print(f"{'='*50}\n")
        best_params = tune_hyperparameters(algo, n_trials=50) 