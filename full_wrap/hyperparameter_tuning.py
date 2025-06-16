import optuna
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, A2C
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

        elif algo_name == "a2c":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
                "n_steps": trial.suggest_int("n_steps", 8, 2048, log=True),
                "ent_coef": trial.suggest_float("ent_coef", 1e-7, 0.1, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
                "rms_prop_eps": trial.suggest_float("rms_prop_eps", 1e-6, 1e-3, log=True),
                "use_rms_prop": trial.suggest_categorical("use_rms_prop", [True, False]),
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

            model = A2C(
                "MlpPolicy",
                env,
                verbose=0,
                device=device,
                learning_rate=params["learning_rate"],
                gamma=params["gamma"],
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                rms_prop_eps=params["rms_prop_eps"],
                use_rms_prop=params["use_rms_prop"],
                policy_kwargs=policy_kwargs,
            )

        elif algo_name == "ppo":
            def get_valid_batch_sizes(n, min_bs=32, max_bs=512):
                return [i for i in range(min_bs, min(n + 1, max_bs + 1)) if n % i == 0]

            n_steps = trial.suggest_int("n_steps", 64, 2048, log=True)
            valid_batch_sizes = get_valid_batch_sizes(n_steps)
            if not valid_batch_sizes:
                valid_batch_sizes = [n_steps]  # safe fallback

            batch_size_idx = trial.suggest_int("batch_size_idx", 0, len(valid_batch_sizes) - 1)
            batch_size = valid_batch_sizes[batch_size_idx]

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": trial.suggest_int("n_epochs", 4, 20),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
                "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
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

            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                device=device,
                learning_rate=params["learning_rate"],
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                n_epochs=params["n_epochs"],
                gamma=params["gamma"],
                clip_range=params["clip_range"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                gae_lambda=params["gae_lambda"],
                policy_kwargs=policy_kwargs,
            )



        else:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

        model.learn(total_timesteps=50000, progress_bar=False)

        mean_reward = 0
        n_eval_episodes = 20
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

    if algo_name == "td3":
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

    if algo_name == "ppo":
        best_known_params = {
            "n_steps": 68,
            "batch_size_idx": 1,
            "learning_rate": 0.000344580194153308,
            "n_epochs": 4,
            "gamma": 0.9479956062170828,
            "clip_range": 0.13195201626331293,
            "ent_coef": 1.0974275204517842e-05,
            "vf_coef": 0.6245891170378073,
            "max_grad_norm": 2.4987564934099336,
            "gae_lambda": 0.9381761664814343,
            "n_layers": 3,
            "layer_size": 249,
            "activation_fn": "tanh",
        }
        study.enqueue_trial(best_known_params)


    # pbar = tqdm(
    #     total=n_trials,
    #     desc=f"Tuning {algo_name.upper()}",
    #     file=sys.stdout,
    #     dynamic_ncols=True,
    #     leave=True,
    # )

    # def update_progress(study, trial):
    #     pbar.update(1)
    #     pbar.refresh()
    #     pbar.set_postfix({"best_value": f"{study.best_value:.2f}"})

    # study.optimize(
    #     lambda trial: objective(trial, algo_name),
    #     n_trials=n_trials,
    #     callbacks=[update_progress],
    #     n_jobs=n_parallel,
    #)

    # pbar.close()

    best_params = study.best_params
    best_value = study.best_value
    results = {
        "best_params": best_params,
        "best_value": best_value,
        "n_trials": n_trials,
        "best_net_arch": [best_params["layer_size"]] * best_params["n_layers"]
        if "layer_size" in best_params else None,
        "best_activation_fn": best_params.get("activation_fn"),
    }

    os.makedirs("hyperparameter_results", exist_ok=True)
    with open(f"hyperparameter_results/{algo_name}_best_params.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nBest parameters for {algo_name}:")
    print(f"Best value: {best_value}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    if results["best_net_arch"]:
        print(f"  net_arch: {results['best_net_arch']}")
    if results["best_activation_fn"]:
        print(f"  activation_fn: {results['best_activation_fn']}")

    return best_params


if __name__ == "__main__":
    algorithms = ["ppo"]
    print("Starting hyperparameter tuning...")
    for algo in algorithms:
        print(f"\n{'=' * 50}")
        print(f"Tuning hyperparameters for {algo.upper()}...")
        print(f"{'=' * 50}\n")
        tune_hyperparameters(algo_name=algo, n_trials=50, n_parallel=4)
