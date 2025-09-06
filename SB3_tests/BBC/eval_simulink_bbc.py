# eval_simulink_bbc.py - Evaluate and visualize a trained agent in the Simulink environment

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, SAC, TD3, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import the custom Simulink environment and the necessary wrapper
from BBCSimulink_env import BBCSimulinkEnv, DiscretizeDutyWrapper

def evaluate_agent_simulink(
    model_path: str,
    stats_path: str,
    algo: str,
    simulink_model: str,
    num_episodes: int = 1,
    dqn_bins: int = 41 
):
    """
    Loads a trained model and evaluates it in the Simulink environment.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    if not os.path.exists(stats_path):
        print(f"Error: VecNormalize stats file not found at {stats_path}")
        return

    # 1. Create the Simulink environment
    def env_fn():
        env = BBCSimulinkEnv(model_name=simulink_model)
        
        if algo.lower() == 'dqn':
            print(f"DQN detected. Wrapping environment with {dqn_bins} discrete actions.")
            env = DiscretizeDutyWrapper(
                env,
                n_bins=dqn_bins,
                low=env.action_space.low[0], 
                high=env.action_space.high[0]
            )
        return env

    env = DummyVecEnv([env_fn])

    # 2. Load the saved normalization statistics
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    print("Simulink environment created and normalization stats loaded.")

    # 3. Load the trained model
    algo_class = {"a2c": A2C, "sac": SAC, "td3": TD3, "dqn": DQN}.get(algo.lower())
    if algo_class is None:
        raise ValueError(f"Unknown algorithm: {algo}")
    model = algo_class.load(model_path, env=env)
    print(f"Model {model_path} loaded successfully.")

    try:
        ep_rewards = []
        ep_lengths = []
        last_ep_data = {"vC": [], "duty_cmd": [], "target_v": [], "iL": [], "time": []}

        # 4. Run the evaluation loop
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            if i == num_episodes - 1:
                last_ep_data = {"vC": [], "duty_cmd": [], "target_v": [], "iL": [], "time": []}

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)

                episode_reward += reward[0]
                episode_length += 1
                
                if i == num_episodes - 1:
                    info = infos[0]
                    # --- FIX IS HERE ---
                    # Get the original, un-normalized observation from the VecNormalize wrapper.
                    # The first element of this is ALWAYS the true, current voltage.
                    original_obs = env.get_original_obs()
                    current_vc = original_obs[0][0]
                    
                    last_ep_data["vC"].append(current_vc)
                    # --- END OF FIX ---

                    last_ep_data["duty_cmd"].append(info.get("duty_cmd", np.nan))
                    last_ep_data["iL"].append(info.get("iL", np.nan))
                    target_v = original_obs[0, -1]
                    last_ep_data["target_v"].append(target_v)
                    current_time = env.envs[0].unwrapped.time if algo.lower() == 'dqn' else env.envs[0].time
                    last_ep_data["time"].append(current_time)

            ep_rewards.append(episode_reward)
            ep_lengths.append(episode_length)
            print(f"Episode {i+1}/{num_episodes} -> Reward: {episode_reward:.2f}, Length: {episode_length}")

        print("\n--- Simulink Evaluation Summary ---")
        print(f"Avg Reward: {np.mean(ep_rewards):.2f} +/- {np.std(ep_rewards):.2f}")
        
        plot_simulink_results(last_ep_data, algo.upper())

    finally:
        print("Closing Simulink environment and MATLAB engine...")
        env.close()
        print("Done.")

def plot_simulink_results(data, algo_name):
    """Generates a plot of voltage, current, and duty cycle vs. time."""
    if not data["vC"]:
        print("No data collected for plotting.")
        return

    time_axis = data["time"]
    
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    color = 'tab:blue'
    ax1.set_ylabel('Output Voltage (V)', color=color)
    ax1.plot(time_axis, data["vC"], color=color, label='Output Voltage (vC)')
    ax1.plot(time_axis, data["target_v"], color='red', linestyle='--', label=f'Target Voltage')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':')
    ax1.set_title('Simulink Model Response')

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Duty Cycle', color=color)
    ax2.plot(time_axis, data["duty_cmd"], color=color, alpha=0.7, label='Duty Cycle')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)

    color = 'tab:purple'
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Inductor Current (A)', color=color)
    ax3.plot(time_axis, data["iL"], color=color, label='Inductor Current (iL)')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.grid(True, linestyle=':')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    fig.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right', bbox_to_anchor=(0.9, 0.9))

    fig.suptitle(f'Agent Performance in Simulink ({algo_name})', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_filename = f"simulink_evaluation_{algo_name.lower()}.png"
    plt.savefig(plot_filename)
    print(f"\n✅ Plot successfully saved to: {plot_filename}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained agent in a Simulink model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model .zip file.")
    parser.add_argument("--stats-path", type=str, required=True, help="Path to the VecNormalize stats .pkl file.")
    parser.add_argument("--algo", choices=["a2c", "sac", "td3", "dqn"], required=True, help="Algorithm used for training.")
    parser.add_argument("--model-name", type=str, default="bbcSim", help="Name of the Simulink model file (e.g., 'bbcSim').")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run (usually 1 is enough for Simulink).")
    parser.add_argument("--dqn-bins", type=int, default=41, help="Number of action bins for DQN (must match training).")

    args = parser.parse_args()

    evaluate_agent_simulink(
        model_path=args.model_path,
        stats_path=args.stats_path,
        algo=args.algo,
        simulink_model=args.model_name,
        num_episodes=args.episodes,
        dqn_bins=args.dqn_bins
    )

