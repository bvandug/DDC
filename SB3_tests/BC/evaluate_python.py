import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, A2C, TD3, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Make sure the environment file is named 'buck_converter_env.py'
# and is in the same directory as this script.
from PYBCEnv import BuckConverterEnv

# --- Import for Google Drive Mounting ---
try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

def plot_and_save_summary(all_episode_data, target_voltage, tolerance, model_type, plot_save_base):
    """
    Plots the voltage vs. time for all evaluation episodes and saves two versions:
    a full view and a zoomed-in view.
    """
    
    # --- Plot 1: Full View ---
    full_view_path = f"{plot_save_base}_{model_type}_full_view.png"
    print(f"\n--- Generating and saving full view plot to {full_view_path} ---")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(15, 8))

    for i, (times, voltages) in enumerate(all_episode_data):
        ax1.plot(times * 1000, voltages, label=f'Episode {i + 1} Voltage', alpha=0.8)

    ax1.axhline(y=target_voltage, color='r', linestyle='--', linewidth=2, label='Target Voltage')
    ax1.axhspan(target_voltage - tolerance, target_voltage + tolerance, color='red', alpha=0.1, label=f'Tolerance (±{tolerance}V)')

    ax1.set_title(f'{model_type} Agent Performance (Target: {target_voltage}V)', fontsize=16, weight='bold')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Output Voltage (V)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if all_episode_data and any(len(v) > 0 for _, v in all_episode_data):
        min_volt = np.min([np.min(v) for _, v in all_episode_data if len(v)>0])
        max_volt = np.max([np.max(v) for _, v in all_episode_data if len(v)>0])
        ax1.set_ylim(bottom=min(target_voltage - 5, min_volt - 1), 
                     top=max(target_voltage + 5, max_volt + 1))
    
    fig1.savefig(full_view_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("--- Full view plot saved successfully! ---")

    # --- Plot 2: Zoomed-in View ---
    zoomed_view_path = f"{plot_save_base}_{model_type}_zoomed_view.png"
    print(f"\n--- Generating and saving zoomed view plot to {zoomed_view_path} ---")
    
    fig2, ax2 = plt.subplots(figsize=(15, 8))

    for i, (times, voltages) in enumerate(all_episode_data):
        ax2.plot(times * 1000, voltages, label=f'Episode {i + 1} Voltage', alpha=0.8, marker='.', markersize=2, linestyle='-')

    ax2.axhline(y=target_voltage, color='r', linestyle='--', linewidth=2, label='Target Voltage')
    ax2.axhspan(target_voltage - tolerance, target_voltage + tolerance, color='red', alpha=0.1, label=f'Tolerance (±{tolerance}V)')

    ax2.set_title(f'{model_type} Agent Performance - Zoomed View (Target: {target_voltage}V)', fontsize=16, weight='bold')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Output Voltage (V)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(bottom=target_voltage - 2.5, top=target_voltage + 2.5)
    
    fig2.savefig(zoomed_view_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("--- Zoomed view plot saved successfully! ---")


def run_evaluation(model, env, n_episodes=5, target_voltage=30.0, tolerance=0.5):
    """
    Evaluates a trained model on the Python environment and returns performance metrics.
    """
    all_rewards, stabilisation_times, stabilization_durations = [], [], []
    steady_state_errors, overshoots, undershoots = [], [], []
    all_episode_plot_data = []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        episode_voltages, episode_times = [], []
        
        print(f"\n--- Running Evaluation Episode {ep + 1}/{n_episodes} ---")

        # Use an infinite loop and break internally to handle the final step correctly
        while True:
            action, _ = model.predict(obs, deterministic=True)
            
            # Capture the state BEFORE taking the step
            unnormalized_obs = env.get_original_obs()
            current_voltage = unnormalized_obs[0][0]
            current_time = env.get_attr('total_sim_time')[0]
            
            # Now, take the step
            obs, reward, dones, infos = env.step(action)
            
            # Log the state that corresponds to the action just taken
            error = current_voltage - target_voltage
            # print(f"Time: {current_time:.6f} s, Voltage: {current_voltage:8.4f} V, Error: {error:8.4f} V, Action: {action[0][0]:.4f}")

            ep_reward += reward[0]
            episode_voltages.append(current_voltage)
            episode_times.append(current_time)
            
            # Break the loop AFTER processing the step if the episode is done
            if dones[0]:
                break

        times_arr = np.array(episode_times)
        voltages_arr = np.array(episode_voltages)

        if len(times_arr) == 0: continue
        
        all_episode_plot_data.append((times_arr, voltages_arr))

        # --- Metric Calculation ---
        has_stabilized = False
        stabilisation_time = times_arr[-1]
        stable_time_for = 0.0
        steady_error = 0.0
        
        outside_tolerance_indices = np.where(np.abs(voltages_arr - target_voltage) > tolerance)[0]
        
        if len(outside_tolerance_indices) == 0:
            first_stable_index = 0
            has_stabilized = True
            stabilisation_time = times_arr[0] if len(times_arr) > 0 else 0
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
        steady_state_errors.append(steady_error if has_stabilized else np.nan)
        overshoots.append(overshoot)
        undershoots.append(undershoot)

        print(f"  Episode Reward         : {ep_reward:.2f}")
        print(f"  Stabilisation Time     : {stabilisation_time * 1000:.2f} ms" if has_stabilized else "  Stabilisation Time     : Not achieved")
        print(f"  Stabilisation Duration : {stable_time_for * 1000:.2f} ms")
        print(f"  Steady-State Error     : {steady_error:+.4f} V" if has_stabilized else "  Steady-State Error     : N/A")
        print(f"  Overshoot              : {overshoot:.4f} V")
        print(f"  Undershoot             : {undershoot:.4f} V")

    print("\n--- Evaluation Summary ---")
    print("-" * 40)
    print(f"Mean Reward             : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean Stabilisation Time : {np.mean(stabilisation_times) * 1000:.2f} ms")
    print(f"Mean Stabilisation Dur. : {np.mean(stabilization_durations) * 1000:.2f} ms")
    with np.errstate(invalid='ignore'): # Suppress warning for mean of empty slice
        print(f"Mean Steady-State Error : {np.nanmean(steady_state_errors):.4f} V")
    print(f"Mean Overshoot          : {np.mean(overshoots):.4f} V")
    print(f"Mean Undershoot         : {np.mean(undershoots):.4f} V")
    print("-" * 40)
    
    return all_episode_plot_data

if __name__ == '__main__':
    # --- Mount Google Drive if running in Colab ---
    if IS_COLAB:
        print("Running in Google Colab. Assuming Drive is already mounted.")
        drive_base_path = "/content/drive/MyDrive/DDC"
    else:
        drive_base_path = "./DDC"

    # --- CONFIGURATION ---
    MODEL_TO_EVALUATE = 'SAC'
    
    # --- CHOOSE EVALUATION MODE ---
    EVALUATE_GENERALIZATION = True # Set to True to test multiple voltages, False for a single voltage.
    
    # --- Settings for Fixed Voltage Evaluation ---
    FIXED_TARGET_VOLTAGE = 30.0 
    
    # --- Settings for Generalization Evaluation ---
    # This list is used only if EVALUATE_GENERALIZATION is True
    GENERALIZATION_VOLTAGES = [27.5, 29.0, 30.0, 31.5, 32.5] 

    # Determine which training mode files to load based on the evaluation type
    if EVALUATE_GENERALIZATION:
        TRAINING_MODE = "Randomized" # Must load a model trained on randomized goals
    else:
        TRAINING_MODE = "Fixed" # Can load a model trained on a fixed goal

    N_EVAL_EPISODES = 1
    MAX_EPISODE_STEPS = 2001

    # --- Determine the list of voltages to test ---
    if EVALUATE_GENERALIZATION:
        evaluation_voltages = GENERALIZATION_VOLTAGES
        print("--- Mode: Generalization Test ---")
    else:
        evaluation_voltages = [FIXED_TARGET_VOLTAGE]
        print(f"--- Mode: Fixed Voltage Test at {FIXED_TARGET_VOLTAGE}V ---")


    # --- Paths for LOADING model files from GOOGLE DRIVE ---
    model_folder_path = os.path.join(drive_base_path, f"{MODEL_TO_EVALUATE}_{TRAINING_MODE}_Python_1")
    MODEL_LOAD_PATH = os.path.join(model_folder_path, f"{MODEL_TO_EVALUATE.lower()}_bc_model_final.zip")
    STATS_LOAD_PATH = os.path.join(model_folder_path, f"{MODEL_TO_EVALUATE.lower()}_vec_normalize_final.pkl")
    
    # --- Main evaluation loop for each target voltage ---
    for target_voltage in evaluation_voltages:
        print("\n" + "="*80)
        print(f"--- STARTING EVALUATION FOR TARGET VOLTAGE: {target_voltage}V ---")
        print("="*80 + "\n")

        # --- Paths for SAVING evaluation results to GOOGLE DRIVE ---
        if EVALUATE_GENERALIZATION:
            plot_save_folder = os.path.join(model_folder_path, "generalization_tests")
        else:
            plot_save_folder = model_folder_path # Save in the main model folder for fixed tests
        
        os.makedirs(plot_save_folder, exist_ok=True)
        PLOT_SAVE_BASE = os.path.join(plot_save_folder, f"eval_at_{target_voltage:.1f}V")

        # --- Setup ---
        print(f"Loading model and stats from Drive path: {model_folder_path}")
        print(f"Saving plots to Drive path:              {plot_save_folder}")
        env = None
        try:
            # The actual evaluation environment with a fixed goal for this run
            env_fn_eval = lambda: BuckConverterEnv(
                render_mode=None,
                use_randomized_goal=False,
                fixed_goal_voltage=target_voltage, # Use the current voltage from the loop
                max_episode_steps=MAX_EPISODE_STEPS
            )
            
            env = DummyVecEnv([env_fn_eval])
            env = VecNormalize.load(STATS_LOAD_PATH, env)
            env.training = False
            env.norm_reward = False
            
            model_class = globals()[MODEL_TO_EVALUATE]
            model = model_class.load(MODEL_LOAD_PATH, env=env)
            
            # --- Run Evaluation ---
            start_time = time.perf_counter()
            plot_data = run_evaluation(
                model=model,
                env=env,
                n_episodes=N_EVAL_EPISODES,
                target_voltage=target_voltage, # Pass the current voltage to the function
                tolerance=0.5
            )
            end_time = time.perf_counter()
            print(f"\nTotal evaluation time for {target_voltage}V: {end_time - start_time:.2f} seconds")

            # --- Generate and Save Summary Plot ---
            if plot_data:
                plot_and_save_summary(
                    all_episode_data=plot_data,
                    target_voltage=target_voltage, # Pass the current voltage
                    tolerance=0.5,
                    model_type=MODEL_TO_EVALUATE,
                    plot_save_base=PLOT_SAVE_BASE # Pass the unique save name
                )

        except FileNotFoundError as e:
            print(f"\n[ERROR] A required file was not found: {e}")
            print("Please ensure the model and stats files exist at the specified Google Drive path.")
            break # Stop if files are not found
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
        finally:
            if env:
                env.close()
                print(f"\n--- Evaluation for {target_voltage}V complete. Environment closed. ---")
