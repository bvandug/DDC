import optuna
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from simulink_env import SimulinkEnv
import json
import os
import torch
from torch import nn
from tqdm import tqdm
import sys


def objective(trial, algo_name):
    env = SimulinkEnv(model_name="PendCart", agent_block="PendCart/RL Agent", dt=0.01)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if algo_name == "td3":
            # Suggested hyperparameters
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
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "layer_size": trial.suggest_int("layer_size", 32, 256),
                "activation_fn": trial.suggest_categorical(
                    "activation_fn", ["tanh", "relu", "leaky_relu", "elu"]
                ),
            }

            net_arch = [params["layer_size"]] * params["n_layers"]
            activation_map = {
                "tanh": nn.Tanh,
                "relu": nn.ReLU,
                "leaky_relu": nn.LeakyReLU,
                "elu": nn.ELU,
            }
            activation_fn = activation_map[params["activation_fn"]]

            policy_kwargs = {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
            }

            model = TD3(
                "MlpPolicy",
                env,
                verbose=0,
                device=device,
                learning_rate=params["learning_rate"],
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                gamma=params["gamma"],
                train_freq=(1, "step"),
                policy_delay=params["policy_delay"],
                action_noise=NormalActionNoise(
                    mean=np.zeros(1),
                    sigma=params["action_noise_sigma"] * np.ones(1),
                ),
                target_policy_noise=params["target_policy_noise"],
                target_noise_clip=params["target_noise_clip"],
                policy_kwargs=policy_kwargs,
            )

        model.learn(total_timesteps=10000, progress_bar=False)

        mean_reward = 0
        n_eval_episodes = 5
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            mean_reward += episode_reward

        return mean_reward / n_eval_episodes

    finally:
        env.close()


def tune_hyperparameters(algo_name, n_trials=50, n_parallel=4):
    print(f"\nTuning {algo_name.upper()} with {n_parallel} parallel workers")

    study = optuna.create_study(direction="maximize")

    # Seed Optuna with known best parameters (from Trial 11)
    best_known_params = {
        "learning_rate": 0.0009531916300780667,
        "buffer_size": 127040,
        "batch_size": 511,
        "tau": 0.007702202735525895,
        "gamma": 0.9428889303152354,
        "policy_delay": 4,
        "action_noise_sigma": 0.1020777897320515,
        "target_policy_noise": 0.10477469465730704,
        "target_noise_clip": 0.5635043295165336,
        "n_layers": 2,
        "layer_size": 64,
        "activation_fn": "relu",
    }
    study.enqueue_trial(best_known_params)

    pbar = tqdm(
        total=n_trials,
        desc=f"Tuning {algo_name.upper()}",
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
    )

    def update_progress(study, trial):
        pbar.update(1)
        pbar.refresh()
        pbar.set_postfix({"best_value": f"{study.best_value:.2f}"})

    study.optimize(
        lambda trial: objective(trial, algo_name),
        n_trials=n_trials,
        callbacks=[update_progress],
        n_jobs=n_parallel,
    )

    pbar.close()

    best_params = study.best_params
    best_value = study.best_value
    results = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": n_trials,
        "best_net_arch": [best_params["layer_size"]] * best_params["n_layers"],
        "best_activation_fn": best_params["activation_fn"],
    }

    os.makedirs("hyperparameter_results", exist_ok=True)
    with open(f"hyperparameter_results/{algo_name}_best_params.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBest parameters for {algo_name}:")
    print(f"Best value: {best_value}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  net_arch: {results['best_net_arch']}")
    print(f"  activation_fn: {results['best_activation_fn']}")

    return best_params


if __name__ == "__main__":
    algorithms = ["td3"]
    print("Starting hyperparameter tuning...")
    for algo in algorithms:
        print(f"\n{'=' * 50}")
        print(f"Tuning hyperparameters for {algo.upper()}...")
        print(f"{'=' * 50}\n")
        tune_hyperparameters(algo_name=algo, n_trials=50, n_parallel=4)
