import numpy as np
import time
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from BCSimulinkEnv import BCSimulinkEnv, DiscretizeActionWrapper
from stable_baselines3 import SAC, TD3, A2C, DDPG, DQN, PPO
from tqdm import tqdm

def plot_and_save_summary(all_episode_data, target_voltage, tolerance, model_type, plot_save_base):
    """
    Plots the voltage vs. time for all evaluation episodes, saves the plots,
    and saves the raw data to a CSV file.

    Args:
        all_episode_data (list): A list of tuples, where each tuple contains (times_arr, voltages_arr).
        target_voltage (float): The target voltage.
        tolerance (float): The voltage tolerance for stabilization.
        model_type (str): The type of the model being evaluated (e.g., 'SAC').
        plot_save_base (str): The base name for the saved plot and data files.
    """
    # --- Save the raw data to a CSV file ---
    data_save_path = f"{plot_save_base}_{target_voltage}_{model_type}_data.csv"
    print(f"\n--- Saving evaluation data to {data_save_path} ---")
    
    # Prepare data for saving by combining all episodes into a single array
    all_data_to_save = []
    for i, (times, voltages) in enumerate(all_episode_data):
        # Create a column for the episode number
        episode_col = np.full((len(times), 1), i + 1)
        # Combine episode number, time (in ms), and voltage into a single row
        episode_data = np.hstack((episode_col, (times * 1000).reshape(-1, 1), voltages.reshape(-1, 1)))
        all_data_to_save.append(episode_data)
        
    # Concatenate all episode data into one array and save to CSV
    if all_data_to_save:
        final_data_array = np.vstack(all_data_to_save)
        np.savetxt(data_save_path, final_data_array, delimiter=',', header='episode,time_ms,voltage_v', comments='')
        print("--- Evaluation data saved successfully! ---")
    else:
        print("--- No data to save. ---")


    # --- Plot the full view graph ---
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
    
    # Dynamic Y-axis limits for the full plot
    min_volt = np.min([np.min(v) for _, v in all_episode_data])
    max_volt = np.max([np.max(v) for _, v in all_episode_data])
    ax1.set_ylim(bottom=min(target_voltage - 5, min_volt - 1), 
                 top=max(target_voltage + 5, max_volt + 1))
    
    fig1.savefig(full_view_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("--- Full view plot saved successfully! ---")

    # --- Plot the Zoomed in view ---
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
    Evaluates a trained model and returns performance metrics and plot data,
    with a single progress bar for total timesteps.    
    """
    all_rewards, stabilisation_times, stabilization_durations = [], [], []
    steady_state_errors, overshoots, undershoots = [], [], []
    ep = 0
    all_episode_plot_data = []

    env.env_method('set_goal_voltage', target_voltage)
    
    # Create a single progress bar instance that will track total timesteps.
    pbar = tqdm(desc="Evaluating Timesteps", unit=" step")
    is_first_step_of_eval = True

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        episode_voltages, episode_times = [], []
        
        while not done:
            # Get the prediction from the model and take a step in the environment
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            
            # Observations are normalized so need to get the unnormalized obs
            unnormalized_obs = env.get_original_obs()
            current_voltage = unnormalized_obs[0][0]
            current_time = env.get_attr('current_time')[0]
            
            # Add the reward to the current and append the voltages and times for plotting
            ep_reward += reward[0]
            episode_voltages.append(current_voltage)
            episode_times.append(current_time)

            # Calculate total number of timesteps for the episode
            if is_first_step_of_eval and len(episode_times) > 1:
                step_time = episode_times[1] - episode_times[0]
                max_time_per_ep = env.get_attr('max_episode_time')[0]
                steps_per_ep = int(max_time_per_ep / step_time) if step_time > 0 else 0
                
                # Set the total for the progress bar
                pbar.total = steps_per_ep * n_episodes
                pbar.refresh() # Display the updated total
                is_first_step_of_eval = False

            # Increment the progress bar on every single timestep
            pbar.update(1)
            
            # Terminate if a termination condition is hit or if the episode is done
            if terminated[0] or info[0].get('TimeLimit.truncated', False):
                done = True

        # Remove the solver artifact for statistic calculations
        times_arr = np.array(episode_times[:-1])
        voltages_arr = np.array(episode_voltages[:-1])

        if len(times_arr) == 0: continue
        
        all_episode_plot_data.append((times_arr, voltages_arr))

        # Metric Calculation
        has_stabilized = False
        stabilisation_time = times_arr[-1]
        stable_time_for = 0.0
        steady_error = 0.0
        
        # Finding the values in voltages array which are outside of the tolerance band
        outside_tolerance_indices = np.where(np.abs(voltages_arr - target_voltage) > tolerance)[0]
        
        # If it immediately stabilises (not feasible)
        if len(outside_tolerance_indices) == 0:
            first_stable_index = 0
            has_stabilized = True
            stabilisation_time = times_arr[0]
        else:
            # Find the index just before it reaches the tolerance band and stabilises there
            last_unstable_index = outside_tolerance_indices[-1]
            if last_unstable_index + 1 < len(times_arr):
                first_stable_index = last_unstable_index + 1
                stabilisation_time = times_arr[first_stable_index]
                has_stabilized = True
        
        if has_stabilized:
            # The steady state voltages are all the voltages in the voltages array from the point where the voltage has stabilised
            steady_state_voltages = voltages_arr[first_stable_index:]
            steady_error = np.mean(steady_state_voltages - target_voltage)
            stable_time_for = times_arr[-1] - stabilisation_time

        # Get the max voltage in the array to calculate the overshoot
        overshoot = max(0, np.max(voltages_arr) - target_voltage)

        # Find where the voltage first hits the tolerance band
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
        steady_state_errors.append(steady_error if has_stabilized else np.nan)
        overshoots.append(overshoot)
        undershoots.append(undershoot)

    pbar.close()


    print(f"--- Episode Results ({target_voltage}V) ---")
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
    # Choose the model to evaluate on MATLAB (A2C, PPO, TD3, DDPG, DQN, SAC)
    MODELS_TO_EVALUATE = ['DQN']
    for MODEL_TO_EVALUATE in MODELS_TO_EVALUATE:
        # Load the model from local directory
        MODEL_SAVE_PATH = f"final_model_{MODEL_TO_EVALUATE}_noise.zip" # The model .zip
        STATS_PATH = f"vec_normalize_{MODEL_TO_EVALUATE}_noise.pkl" # The normalisation weights for the NNs

        # Save the results of the evaluation into google drive
        drive_save_path = f"/content/drive/MyDrive/DDC/{MODEL_TO_EVALUATE}_Noise_Long_Episode/"
        os.makedirs(drive_save_path, exist_ok=True) # Ensure the save directory exists
        PLOT_SAVE_BASE = os.path.join(drive_save_path, "evaluation_results")

        TARGET_VOLTAGE = 30.0 # For fixed voltage testing
        EVALUATION_VOLTAGES = [30.0] # For variable voltage testing
        N_EVAL_EPISODES = 1 # Episodes are determinstic so only need 1
        EVAL_EPISODE_TIME = 0.1 # Episode time 0.1 = 2000 timesteps, 0.3 = 6000 timesteps

        # Setup the evaluation
        for voltage in EVALUATION_VOLTAGES:
            print(f"--- Setting up for STATISTICAL evaluation of {MODEL_TO_EVALUATE} model for {voltage}V ---")
            env = None
            try:
                # Base environment creation function
                def base_env_fn():
                    return BCSimulinkEnv(model_name="bcSim", enable_plotting=False, fixed_goal_voltage=voltage, max_episode_time=EVAL_EPISODE_TIME)

                # If the model is DQN, wrap the environment to handle the discrete action space
                if MODEL_TO_EVALUATE == 'DQN':
                    env_fn = lambda: DiscretizeActionWrapper(base_env_fn(), n_bins=17)
                else:
                    # For other models, use the standard continuous environment
                    env_fn = base_env_fn
                
                # Create the BC environment and wrap it in a dummy vec env and load the normalisation statistics
                env = DummyVecEnv([env_fn])
                env = VecNormalize.load(STATS_PATH, env)
                env.training = False # Set the training to False so SB3 knows to evaluate
                env.norm_reward = False
                
                # Dynamically load the correct model class
                if MODEL_TO_EVALUATE == 'SAC':
                    model = SAC.load(MODEL_SAVE_PATH, env=env)
                elif MODEL_TO_EVALUATE == 'TD3':
                    model = TD3.load(MODEL_SAVE_PATH, env=env)
                elif MODEL_TO_EVALUATE == 'A2C':
                    model = A2C.load(MODEL_SAVE_PATH, env=env)
                elif MODEL_TO_EVALUATE == 'DDPG':
                    model = DDPG.load(MODEL_SAVE_PATH, env=env)
                elif MODEL_TO_EVALUATE == 'PPO':
                    model = PPO.load(MODEL_SAVE_PATH, env=env)
                elif MODEL_TO_EVALUATE == 'DQN':
                    model = DQN.load(MODEL_SAVE_PATH, env=env)
                else:
                    raise ValueError(f"Model type '{MODEL_TO_EVALUATE}' not recognized.")

                
                # Run the evaluation (get the start and end time)
                start_time = time.perf_counter()
                metrics, plot_data = run_evaluation(model=model, env=env, n_episodes=N_EVAL_EPISODES, target_voltage=voltage, tolerance=0.5)
                end_time = time.perf_counter()
                print(f"\nTotal evaluation time: {end_time - start_time:.2f} seconds")

                # Generate and save the plots of the evaluation episode
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
                print(f"Please ensure the model and stats files are in the same directory as the script.")
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
            finally:
                if env:
                    env.close()