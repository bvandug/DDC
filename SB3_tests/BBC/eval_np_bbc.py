# eval_np_bbc.py - Evaluate and visualize a trained agent in the NumPy environment

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, SAC, TD3, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# We need the make_env function and wrappers from the training script
# to ensure the environment is configured identically.
from jax_bbc_train import make_env

def evaluate_agent(
    model_path: str,
    stats_path: str,
    algo: str,
    num_episodes: int = 5,
    k_macro: int = 1,
    dqn_bins: int = 41
):
    """
    Loads a trained model and evaluates it in the environment.

    Args:
        model_path: Path to the trained model's .zip file.
        stats_path: Path to the VecNormalize statistics .pkl file.
        algo: The algorithm name (a2c, sac, td3, dqn).
        num_episodes: How many episodes to run for evaluation.
        k_macro: The macro-step value used during training.
        dqn_bins: The number of duty bins if the model is DQN.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(stats_path):
        print(f"Error: VecNormalize stats file not found at {stats_path}")
        return

    # 1. Create the environment using the same factory function from training
    #    This ensures all wrappers and parameters are identical.
    env_fn = make_env(
        seed=int(time.time()), # Use a different seed for evaluation
        rank=0,
        k_macro=k_macro,
        algo_name=algo,
        dqn_bins=dqn_bins
    )
    env = DummyVecEnv([env_fn])

    # 2. Load the saved normalization statistics
    #    IMPORTANT: Set training=False and norm_reward=False for evaluation.
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    print("Environment created and normalization stats loaded.")

    # 3. Load the trained model
    algo_class = {"a2c": A2C, "sac": SAC, "td3": TD3, "dqn": DQN}.get(algo.lower())
    if algo_class is None:
        raise ValueError(f"Unknown algorithm: {algo}")

    model = algo_class.load(model_path, env=env)
    print(f"Model {model_path} loaded successfully.")

    # --- Data collection for plotting ---
    ep_rewards = []
    ep_lengths = []
    # We will only plot the last episode's data for clarity
    last_ep_data = {"vC": [], "duty_cmd": [], "target_v": []}

    # 4. Run the evaluation loop
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Clear data for the new episode
        if i == num_episodes - 1:
            last_ep_data = {"vC": [], "duty_cmd": [], "target_v": []}

        while not done:
            # Use deterministic=True for a consistent evaluation of the policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            episode_reward += reward[0]
            episode_length += 1
            
            # Store data for the last episode
            if i == num_episodes - 1:
                info = infos[0] # Get info from the single environment
                last_ep_data["vC"].append(info.get("vC", np.nan))
                last_ep_data["duty_cmd"].append(info.get("duty_cmd", np.nan))
                # The target voltage is the last element of the observation vector
                # We need to un-normalize it to get the real value.
                target_v = env.get_original_obs()[0, -1]
                last_ep_data["target_v"].append(target_v)

        ep_rewards.append(episode_reward)
        ep_lengths.append(episode_length)
        print(f"Episode {i+1}/{num_episodes} -> Reward: {episode_reward:.2f}, Length: {episode_length}")

    print("\n--- Evaluation Summary ---")
    print(f"Avg Reward: {np.mean(ep_rewards):.2f} +/- {np.std(ep_rewards):.2f}")
    print(f"Avg Length: {np.mean(ep_lengths):.2f} +/- {np.std(ep_lengths):.2f}")
    
    # 5. Plot the results of the last episode
    plot_results(last_ep_data, algo.upper())


def plot_results(data, algo_name):
    """Generates a plot of voltage and duty cycle vs. time steps."""
    if not data["vC"]:
        print("No data collected for plotting.")
        return

    timesteps = np.arange(len(data["vC"]))
    
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Voltage on the primary Y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Output Voltage (V)', color=color)
    ax1.plot(timesteps, data["vC"], color=color, label='Output Voltage (vC)')
    ax1.plot(timesteps, data["target_v"], color='red', linestyle='--', label=f'Target Voltage ({data["target_v"][0]} V)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':')

    # Create a secondary Y-axis for Duty Cycle
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Duty Cycle', color=color)
    ax2.plot(timesteps, data["duty_cmd"], color=color, alpha=0.7, label='Duty Cycle')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)

    fig.suptitle(f'Agent Performance ({algo_name})', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained buck-boost converter agent.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model .zip file.")
    parser.add_argument("--stats-path", type=str, required=True, help="Path to the VecNormalize stats .pkl file.")
    parser.add_argument("--algo", choices=["a2c", "sac", "td3", "dqn"], required=True, help="Algorithm used for training.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run for evaluation.")
    # Pass the same geometry/wrapper args used in training
    parser.add_argument("--macro-k", type=int, default=1, help="Repeat each action for k PWM periods (must match training).")
    parser.add_argument("--dqn-bins", type=int, default=41, help="Number of duty bins for DQN (must match training).")

    args = parser.parse_args()

    evaluate_agent(
        model_path=args.model_path,
        stats_path=args.stats_path,
        algo=args.algo,
        num_episodes=args.episodes,
        k_macro=args.macro_k,
        dqn_bins=args.dqn_bins
    )