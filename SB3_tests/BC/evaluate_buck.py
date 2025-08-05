import numpy as np
import time
import os
import matplotlib.pyplot as plt
import gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from BCSimulinkEnv import BCSimulinkEnv
from stable_baselines3 import SAC, TD3, A2C, DDPG, DQN, PPO
from tqdm import tqdm

# --- The DiscretizeActionWrapper class for DQN ---
class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    A wrapper to discretize a continuous action space for DQN.
    """
    def __init__(self, env, n_bins=17):
        super().__init__(env)
        self.n_bins = n_bins
        # The action space is now an integer from 0 to n_bins-1
        self.action_space = gym.spaces.Discrete(self.n_bins)
        # Create the mapping from the integer action to a continuous value
        self.continuous_actions = np.linspace(
            self.env.action_space.low[0],
            self.env.action_space.high[0],
            self.n_bins
        )

    def action(self, action):
        """
        Translates the discrete action from the agent into its
        corresponding continuous value for the underlying environment.
        """
        # Select the continuous value from the map
        continuous_action = self.continuous_actions[action]
        # Return it in the shape the environment expects
        return np.array([continuous_action], dtype=np.float32)


def plot_and_save_summary(all_episode_data, target_voltage, tolerance, model_type, plot_save_base):
    """
    Plots the voltage vs. time for all evaluation episodes and saves two versions:
    a full view and a zoomed-in view.
    """
    # Plot the full view graph
    full_view_path = f"{plot_save_base}_{target_voltage}_{model_type}_full_view.png"
    print(f"\n--- Generating and saving full view plot to {full_view_path} ---")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(15, 8))

    for i, (times, voltages) in enumerate(all_episode_data):
        ax1.plot(times * 1000, voltages, label=f'Episode {i + 1} Voltage', alpha=0.8)

    ax1.axhline(y=target_voltage, color='r', linestyle='--', linewidth=2, label='Target Voltage')
    ax1.axhspan(target_voltage - tolerance, target_voltage + tolerance, color='red', alpha=0.1, label=f'Tolerance (±{tolerance}V)')

    ax1.set_title(f'{model_type} Agent Performance Over Evaluation Episodes', fontsize=16, weight='bold')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Output Voltage (V)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    min_volt = np.min([np.min(v) for _, v in all_episode_data])
    max_volt = np.max([np.max(v) for _, v in all_episode_data])
    ax1.set_ylim(bottom=min(target_voltage - 5, min_volt - 1), 
                 top=max(target_voltage + 5, max_volt + 1))
    
    fig1.savefig(full_view_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("--- Full view plot saved successfully! ---")

    # Plot the Zoomed in view
    zoomed_view_path = f"{plot_save_base}_{target_voltage}_{model_type}_zoomed_view.png"
    print(f"\n--- Generating and saving zoomed view plot to {zoomed_view_path} ---")
    
    fig2, ax2 = plt.subplots(figsize=(15, 8))

    for i, (times, voltages) in enumerate(all_episode_data):
        ax2.plot(times * 1000, voltages, label=f'Episode {i + 1} Voltage', alpha=0.8, marker='.', linestyle='-')

    ax2.axhline(y=target_voltage, color='r', linestyle='--', linewidth=2, label='Target Voltage')
    ax2.axhspan(target_voltage - tolerance, target_voltage + tolerance, color='red', alpha=0.1, label=f'Tolerance (±{tolerance}V)')

    ax2.set_title(f'{model_type} Agent Performance (Zoomed View)', fontsize=16, weight='bold')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Output Voltage (V)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(bottom=target_voltage - 2.5, top=target_voltage + 2.5)
    
    fig2.savefig(zoomed_view_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("--- Zoomed view plot saved successfully! ---")


def run_evaluation(model, env, n_episodes=1, target_voltage=30.0, tolerance=0.5):
    """
    Evaluates a trained model and returns performance metrics and plot data.
    """
    all_rewards, stabilisation_times, stabilization_durations = [], [], []
    steady_state_errors, overshoots, undershoots = [], [], []
    all_episode_plot_data = []

    env.env_method('set_goal_voltage', target_voltage)
    
    pbar = tqdm(desc="Evaluating Timesteps", unit=" step", leave=False)
    is_first_step_of_eval = True

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        episode_voltages, episode_times = [], []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            
            unnormalized_obs = env.get_original_obs()
            current_voltage = unnormalized_obs[0][0]
            current_time = env.get_attr('current_time')[0]
            
            ep_reward += reward[0]
            episode_voltages.append(current_voltage)
            episode_times.append(current_time)

            if is_first_step_of_eval and len(episode_times) > 1:
                step_time = episode_times[1] - episode_times[0]
                max_time_per_ep = env.get_attr('max_episode_time')[0]
                steps_per_ep = int(max_time_per_ep / step_time) if step_time > 0 else 0
                pbar.total = steps_per_ep * n_episodes
                pbar.refresh()
                is_first_step_of_eval = False

            pbar.update(1)
            
            if terminated[0] or info[0].get('TimeLimit.truncated', False):
                done = True

        times_arr = np.array(episode_times[:-1])
        voltages_arr = np.array(episode_voltages[:-1])

        if len(times_arr) == 0: continue
        all_episode_plot_data.append((times_arr, voltages_arr))

        # --- Metric Calculation ---
        has_stabilized = False
        stabilisation_time = times_arr[-1]
        stable_time_for = 0.0
        
        outside_tolerance_indices = np.where(np.abs(voltages_arr - target_voltage) > tolerance)[0]
        
        if len(outside_tolerance_indices) == 0:
            first_stable_index = 0
            has_stabilized = True
            stabilisation_time = times_arr[0]
        else:
            last_unstable_index = outside_tolerance_indices[-1]
            if last_unstable_index + 1 < len(times_arr):
                first_stable_index = last_unstable_index + 1
                stabilisation_time = times_arr[first_stable_index]
                has_stabilized = True
        
        if has_stabilized:
            steady_state_voltages = voltages_arr[first_stable_index:]
            steady_error = np.mean(steady_state_voltages - target_voltage)
            stable_time_for = times_arr[-1] - stabilisation_time
        else:
            steady_error = np.nan

        overshoot = max(0, np.max(voltages_arr) - target_voltage)
        
        first_cross_indices = np.where(voltages_arr >= target_voltage)[0]
        if len(first_cross_indices) > 0:
            first_cross_index = first_cross_indices[0]
            voltages_after_cross = voltages_arr[first_cross_index:]
            undershoot = max(0, target_voltage - np.min(voltages_after_cross))
        else:
            undershoot = 0.0

        all_rewards.append(ep_reward)
        stabilisation_times.append(stabilisation_time)
        stabilization_durations.append(stable_time_for)
        steady_state_errors.append(steady_error)
        overshoots.append(overshoot)
        undershoots.append(undershoot)

    pbar.close()

    print(f"\n--- Episode Results ({target_voltage}V) ---")
    print("-" * 40)
    print(f"Mean Reward             : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean Stabilisation Time : {np.mean(stabilisation_times) * 1000:.2f} ms")
    print(f"Mean Stabilisation Dur. : {np.mean(stabilization_durations) * 1000:.2f} ms")
    print(f"Mean Steady-State Error : {np.nanmean(steady_state_errors):.4f} V")
    print(f"Mean Overshoot          : {np.mean(overshoots):.4f} V")
    print(f"Mean Undershoot         : {np.mean(undershoots):.4f} V")
    print("-" * 40)

    metrics = { "rewards": all_rewards, "stabilisation_times": stabilisation_times, "stabilization_durations": stabilization_durations, "steady_state_errors": steady_state_errors, "overshoots": overshoots, "undershoots": undershoots }
    
    return metrics, all_episode_plot_data

if __name__ == '__main__':
    # Choose the model to evaluate (A2C, PPO, TD3, DDPG, DQN, SAC)
    MODEL_TO_EVALUATE = 'DQN'

    MODEL_SAVE_PATH = f"final_model_{MODEL_TO_EVALUATE}.zip"
    STATS_PATH = f"vec_normalize_{MODEL_TO_EVALUATE}.pkl"

    drive_save_path = f"/content/drive/MyDrive/DDC/{MODEL_TO_EVALUATE}/"
    os.makedirs(drive_save_path, exist_ok=True)
    PLOT_SAVE_BASE = os.path.join(drive_save_path, "evaluation_results")

    EVALUATION_VOLTAGES = [27.5, 29.0, 30.0, 31.5, 32.5]
    N_EVAL_EPISODES = 1
    EVAL_EPISODE_TIME = 0.1

    for voltage in EVALUATION_VOLTAGES:
        print(f"\n--- Setting up evaluation of {MODEL_TO_EVALUATE} for {voltage}V ---")
        env = None
        try:
            # Conditionally wrap the environment for DQN
            if MODEL_TO_EVALUATE == 'DQN':
                env_fn = lambda: DiscretizeActionWrapper(
                    BCSimulinkEnv(
                        model_name="bcSim",
                        enable_plotting=False,
                        fixed_goal_voltage=voltage, # Corrected parameter name
                        max_episode_time=EVAL_EPISODE_TIME
                    )
                )
            else:
                # For all other models, use the standard continuous environment
                env_fn = lambda: BCSimulinkEnv(
                    model_name="bcSim",
                    enable_plotting=False,
                    fixed_goal_voltage=voltage, # Corrected parameter name
                    max_episode_time=EVAL_EPISODE_TIME
                )
            
            env = DummyVecEnv([env_fn])
            env = VecNormalize.load(STATS_PATH, env)
            env.training = False
            env.norm_reward = False
            
            # Dynamically load the correct model class
            model_class = globals()[MODEL_TO_EVALUATE]
            model = model_class.load(MODEL_SAVE_PATH, env=env)
            
            start_time = time.perf_counter()
            metrics, plot_data = run_evaluation(
                model=model,
                env=env,
                n_episodes=N_EVAL_EPISODES,
                target_voltage=voltage,
                tolerance=0.5
            )
            end_time = time.perf_counter()
            print(f"Total evaluation time: {end_time - start_time:.2f} seconds")

            if plot_data:
                plot_and_save_summary(
                    all_episode_data=plot_data,
                    target_voltage=voltage,
                    tolerance=0.5,
                    model_type=MODEL_TO_EVALUATE,
                    plot_save_base=PLOT_SAVE_BASE
                )

        except FileNotFoundError as e:
            print(f"\n[ERROR] A required file was not found: {e}")
            print("Please ensure the model and stats files exist.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
        finally:
            if env:
                env.close()