import os
import json
import argparse
import numpy as np
import torch.nn as nn

from simulink_env import SimulinkEnv
from stable_baselines3 import TD3, A2C
from stable_baselines3.common.noise import NormalActionNoise

# Optional: mapping for activation functions
activation_fn_map = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU
}

def load_hyperparameters(algo_name):
    path = f"hyperparameter_results/{algo_name}_best_params.json"
    with open(path, "r") as f:
        config = json.load(f)
    return config["best_params"]

def create_policy_kwargs(params):
    return dict(
        net_arch=[params["layer_size"]] * params["n_layers"],
        activation_fn=activation_fn_map[params["activation_fn"].lower()]
    )

def main(algo_name="td3", timesteps=50000):
    assert algo_name in ["td3", "a2c"], "Algorithm must be 'td3' or 'a2c'"

    # Load the environment
    env = SimulinkEnv(model_name="PendCart", agent_block="PendCart/RL Agent", dt=0.01)

    # Load hyperparameters and build policy
    params = load_hyperparameters(algo_name)
    policy_kwargs = create_policy_kwargs(params)

    # Set model save path
    model_path = f"best_model_{algo_name}"

    print(f"\n=== Training or Resuming {algo_name.upper()} with loaded config ===")

    if algo_name == "td3":
        Algo = TD3
        action_noise = NormalActionNoise(
            mean=np.zeros(1),
            sigma=params["action_noise_sigma"] * np.ones(1)
        )

        if os.path.exists(model_path + ".zip"):
            print(f"Loading existing TD3 model from {model_path}.zip...")
            model = Algo.load(model_path, env=env, action_noise=action_noise)
        else:
            print("No existing TD3 model found. Creating new one...")
            model = Algo(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=params["learning_rate"],
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                gamma=params["gamma"],
                train_freq=(1, "step"),
                policy_delay=params["policy_delay"],
                action_noise=action_noise,
                target_policy_noise=params["target_policy_noise"],
                target_noise_clip=params["target_noise_clip"],
                policy_kwargs=policy_kwargs
            )

    elif algo_name == "a2c":
        Algo = A2C

        if os.path.exists(model_path + ".zip"):
            print(f"Loading existing A2C model from {model_path}.zip...")
            model = Algo.load(model_path, env=env, device="cpu")
        else:
            print("No existing A2C model found. Creating new one...")
            model = Algo(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=params["learning_rate"],
                gamma=params["gamma"],
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                rms_prop_eps=params["rms_prop_eps"],
                use_rms_prop=params["use_rms_prop"],
                policy_kwargs=policy_kwargs,
                device="cpu"
            )


    # Train
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    # Save
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    env.close()
    print("Training complete. Environment closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["td3", "a2c", "ppo"], default="td3", help="Algorithm to train (td3 or a2c)")
    parser.add_argument("--timesteps", type=int, default=50000, help="Number of training timesteps")
    args = parser.parse_args()

    main(algo_name=args.algo, timesteps=args.timesteps)
