import optuna
import numpy as np
from stable_baselines3 import SAC, A2C, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from BCSimulinkEnv import BCSimulinkEnv # Use your primary training environment
import json
import os
import torch
from torch import nn
from tqdm import tqdm
import sys

# --- CONSTANTS FOR TUNING ---
# Total timesteps per trial. A trial is one full run with a set of hyperparameters.
TOTAL_TIMESTEPS_PER_TRIAL = 40000
# How often to check the performance and potentially prune the trial.
EVAL_INTERVAL = 10000
# Number of episodes to average over for each evaluation check.
N_EVAL_EPISODES = 5
# If a trial's reward is below this at a certain timestep, it's a hard fail.
# Format: {timestep: min_reward_required}
HARD_FAIL_THRESHOLDS = {20000: 150, 30000: 300}
# Pruning configuration
MIN_RESOURCES = 10000  # The first check happens at 10k steps.

def objective(trial, algo_name: str):
    """
    The objective function for one Optuna trial.
    It samples hyperparameters, creates an agent, trains it in chunks,
    evaluates its performance, and allows for early stopping (pruning).
    """
    
    # Use a shorter episode time for faster trials
    env_fn = lambda: BCSimulinkEnv(model_name="bcSim", enable_plotting=False, max_episode_time=0.1)
    
    # Each trial needs its own environment
    env = DummyVecEnv([env_fn])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- HYPERPARAMETER SAMPLING ---
        if algo_name.lower() == "sac":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "tau": trial.suggest_float("tau", 0.005, 0.02),
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "layer_size": trial.suggest_int("layer_size", 64, 256),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            policy_kwargs = {"net_arch": dict(pi=net_arch, qf=net_arch)}
            model = SAC("MlpPolicy", env, verbose=0, device=device,
                        learning_rate=params["learning_rate"], gamma=params["gamma"], 
                        tau=params["tau"], policy_kwargs=policy_kwargs)

        elif algo_name.lower() == "a2c":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.90, 0.999),
                "n_steps": trial.suggest_int("n_steps", 8, 2048, log=True),
                "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.01, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.2, 0.8),
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "layer_size": trial.suggest_int("layer_size", 64, 256),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            policy_kwargs = {"net_arch": [dict(pi=net_arch, vf=net_arch)]}
            model = A2C("MlpPolicy", env, verbose=0, device=device,
                        learning_rate=params["learning_rate"], gamma=params["gamma"], 
                        n_steps=params["n_steps"], ent_coef=params["ent_coef"], 
                        vf_coef=params["vf_coef"], policy_kwargs=policy_kwargs)

        elif algo_name.lower() == "td3":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "tau": trial.suggest_float("tau", 0.005, 0.02),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "action_noise_sigma": trial.suggest_float("action_noise_sigma", 1e-3, 0.1, log=True),
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "layer_size": trial.suggest_int("layer_size", 64, 400),
            }
            net_arch = [params["layer_size"]] * params["n_layers"]
            policy_kwargs = {"net_arch": net_arch}
            model = TD3("MlpPolicy", env, verbose=0, device=device,
                        learning_rate=params["learning_rate"], tau=params["tau"], 
                        gamma=params["gamma"],
                        action_noise=NormalActionNoise(mean=np.zeros(1), sigma=params["action_noise_sigma"] * np.ones(1)),
                        policy_kwargs=policy_kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        # --- TRAIN-EVAL LOOP WITH PRUNING ---
        timesteps = 0
        final_reward = 0
        while timesteps < TOTAL_TIMESTEPS_PER_TRIAL:
            model.learn(EVAL_INTERVAL, reset_num_timesteps=False, progress_bar=False)
            timesteps += EVAL_INTERVAL

            # Evaluate the model
            mean_reward = 0
            for _ in range(N_EVAL_EPISODES):
                obs, done, ep_rew = env.reset(), False, 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action)
                    ep_rew += reward
                mean_reward += ep_rew
            mean_reward /= N_EVAL_EPISODES
            final_reward = mean_reward

            # 1. Hard-fail gate
            if timesteps in HARD_FAIL_THRESHOLDS and mean_reward < HARD_FAIL_THRESHOLDS[timesteps]:
                print(f"[{algo_name.upper()}|T{trial.number}] ✘ Hard-fail (<{HARD_FAIL_THRESHOLDS[timesteps]}) at {timesteps} (reward {mean_reward:.1f})")
                raise optuna.TrialPruned()

            # 2. Report intermediate value to the pruner
            trial.report(mean_reward, step=timesteps)
            if trial.should_prune():
                print(f"[{algo_name.upper()}|T{trial.number}] ✘ Pruned by Halving at {timesteps} (reward {mean_reward:.1f})")
                raise optuna.TrialPruned()
        
        return final_reward

    finally:
        # Crucially, close the environment to shut down the MATLAB engine
        env.close()


def tune_hyperparameters(algo_name, n_trials=50):
    """
    Main function to set up and run the Optuna study with a pruner.
    """
    study_name = f"{algo_name}-bc-tuning-advanced"
    storage_name = f"sqlite:///{study_name}.db"
    
    print(f"\nAdvanced Tuning for {algo_name.upper()}. Study: {study_name}")
    print(f"Results will be saved to: {storage_name}")

    # Set up the pruner
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=MIN_RESOURCES)

    study = optuna.create_study(
        study_name=study_name, storage=storage_name,
        load_if_exists=True, direction="maximize", pruner=pruner
    )

    # Use a progress bar for the optimization loop
    pbar = tqdm(total=n_trials, desc=f"Tuning {algo_name.upper()}", file=sys.stdout)
    def _pbar_callback(study, trial):
        pbar.update(1)
        try:
            pbar.set_postfix(best_val=f"{study.best_value:.1f}")
        except ValueError:
            pbar.set_postfix(best_val="N/A")

    study.optimize(
        lambda trial: objective(trial, algo_name),
        n_trials=n_trials,
        n_jobs=1,  # IMPORTANT: Must be 1 for Simulink/MATLAB
        callbacks=[_pbar_callback]
    )
    pbar.close()

    # --- Save and Print Results ---
    results = {"best_value": study.best_value, "best_params": study.best_params}
    results_dir = "hyperparameter_results_advanced"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{algo_name}_best_params.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n--- Best parameters for {algo_name} ---")
    print(f"Best value (mean reward): {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params


if __name__ == "__main__":
    # --- CHOOSE THE MODEL TO TUNE ---
    # Options: "SAC", "A2C", "TD3"
    ALGORITHM_TO_TUNE = "A2C" 
    
    print(f"{'=' * 50}")
    print(f"Advanced hyperparameter tuning for {ALGORITHM_TO_TUNE.upper()}...")
    print(f"{'=' * 50}\n")
    
    tune_hyperparameters(
        algo_name=ALGORITHM_TO_TUNE,
        n_trials=50 # Number of different hyperparameter combinations to try
    )
