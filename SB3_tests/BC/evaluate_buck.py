import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

# Import the Simulink-based environment
from BCSimulinkEnv import BCSimulinkEnv

def plot_and_save_summary(all_episode_data, target_voltage, tolerance, model_type, plot_save_base):
    """
    Plots the voltage vs. time for all evaluation episodes and saves two versions:
    a full view and a zoomed-in view.
    """
    # --- Plot 1: Full View ---
    full_view_path = f"{plot_save_base}_full_view.png"
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
    zoomed_view_path = f"{plot_save_base}_zoomed_view.png"
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
    Evaluates a trained model on the Simulink environment and returns performance metrics.
    """
    all_rewards, stabilisation_times, stabilization_durations = [], [], []
    steady_state_errors, overshoots, undershoots = [], [], []
    all_episode_plot_data = []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        episode_voltages, episode_times = [], []
        
        print(f"\n--- Running Evaluation Episode {ep + 1}/{n_episodes} ---")

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            
            current_voltage = infos[0]['obs'][0]
            current_time = infos[0]['time']
            
            ep_reward += reward[0]
            episode_voltages.append(current_voltage)
            episode_times.append(current_time)

        times_arr = np.array(episode_times)
        voltages_arr = np.array(episode_voltages)

        if len(times_arr) == 0: continue
        
        all_episode_plot_data.append((times_arr, voltages_arr))

        has_stabilized = False
        stabilisation_time, stable_time_for, steady_error = times_arr[-1], 0.0, 0.0
        outside_tolerance_indices = np.where(np.abs(voltages_arr - target_voltage) > tolerance)[0]
        
        if not outside_tolerance_indices.any():
            first_stable_index = 0
            has_stabilized = True
            stabilisation_time = times_arr[0] if len(times_arr) > 0 else 0
        else:
            last_unstable_index = outside_tolerance_indices[-1]
            if last_unstable_index + 1 < len(times_arr):
                first_stable_index, has_stabilized = last_unstable_index + 1, True
                stabilisation_time = times_arr[first_stable_index]
        
        if has_stabilized:
            steady_state_voltages = voltages_arr[first_stable_index:]
            steady_error = np.mean(steady_state_voltages - target_voltage)
            stable_time_for = times_arr[-1] - stabilisation_time

        overshoot = max(0, np.max(voltages_arr) - target_voltage)
        
        first_cross_indices = np.where(voltages_arr >= target_voltage)[0]
        undershoot = max(0, target_voltage - np.min(voltages_arr[first_cross_indices[0]:])) if first_cross_indices.any() else 0.0

        all_rewards.append(ep_reward)
        stabilisation_times.append(stabilisation_time)
        stabilization_durations.append(stable_time_for)
        steady_state_errors.append(steady_error if has_stabilized else np.nan)
        overshoots.append(overshoot)
        undershoots.append(undershoot)

        print(f"  Episode Reward         : {ep_reward:.2f}")
        print(f"  Stabilisation Time     : {stabilisation_time * 1000:.2f} ms" if has_stabilized else "  Stabilisation Time     : Not achieved")
        print(f"  Steady-State Error     : {steady_error:+.4f} V" if has_stabilized else "  Steady-State Error     : N/A")
        print(f"  Overshoot              : {overshoot:.4f} V")
        print(f"  Undershoot             : {undershoot:.4f} V")

    print("\n--- Evaluation Summary ---")
    print("-" * 40)
    print(f"Mean Reward             : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean Stabilisation Time : {np.mean(stabilisation_times) * 1000:.2f} ms")
    with np.errstate(invalid='ignore'):
        print(f"Mean Steady-State Error : {np.nanmean(steady_state_errors):.4f} V")
    print(f"Mean Overshoot          : {np.mean(overshoots):.4f} V")
    print(f"Mean Undershoot         : {np.mean(undershoots):.4f} V")
    print("-" * 40)
    
    return all_episode_plot_data

if __name__ == '__main__':
    # --- MODIFIED: List of the top 3 A2C model folders to evaluate ---
    MODELS_TO_EVALUATE = [
        "A2C_Trial27_Seed42_Best_SO_FAR",
        "A2C_Trial7_Seed42"
    ]
    
    BASE_MODEL_DIR = "./DDC_RL_A2C"
    EVALUATION_SAVE_DIR = "./evaluation_results"

    # --- CONFIGURATION ---
    GENERALIZATION_VOLTAGES = [25, 27.5, 29.0, 30.0, 31.5, 32.5, 35] 
    N_EVAL_EPISODES = 1
    EVAL_EPISODE_TIME = 0.01

    # --- Main loop to iterate through each model folder ---
    for model_folder_name in MODELS_TO_EVALUATE:
        print("\n" + "#"*80)
        print(f"# STARTING EVALUATION FOR MODEL: {model_folder_name}")
        print("#"*80 + "\n")

        training_run_folder = os.path.join(BASE_MODEL_DIR, model_folder_name)
        
        MODEL_LOAD_PATH = os.path.join(training_run_folder, "model_final.zip")
        STATS_LOAD_PATH = os.path.join(training_run_folder, "vec_normalize_final.pkl")
        
        # --- Loop to evaluate each target voltage for the current model ---
        for target_voltage in GENERALIZATION_VOLTAGES:
            print("\n" + "="*80)
            print(f"--- EVALUATING {model_folder_name} FOR TARGET VOLTAGE: {target_voltage}V ---")
            print("="*80 + "\n")

            plot_save_folder = os.path.join(EVALUATION_SAVE_DIR, model_folder_name)
            os.makedirs(plot_save_folder, exist_ok=True)
            PLOT_SAVE_BASE = os.path.join(plot_save_folder, f"eval_at_{target_voltage:.1f}V")

            print(f"Loading model and stats from: {training_run_folder}")
            print(f"Saving plots to:              {plot_save_folder}")
            env = None
            try:
                # Use BCSimulinkEnv for evaluation
                env_fn_eval = lambda: BCSimulinkEnv(
                    model_name='bcSim',
                    enable_plotting=False,
                    fixed_goal_voltage=target_voltage,
                    max_episode_time=EVAL_EPISODE_TIME
                )
                
                env = DummyVecEnv([env_fn_eval])
                env = VecNormalize.load(STATS_LOAD_PATH, env)
                env.training = False
                env.norm_reward = False
                
                # The model class is always A2C for this script
                model = A2C.load(MODEL_LOAD_PATH, env=env)
                
                start_time = time.perf_counter()
                plot_data = run_evaluation(
                    model=model, env=env, n_episodes=N_EVAL_EPISODES,
                    target_voltage=target_voltage, tolerance=0.5
                )
                end_time = time.perf_counter()
                print(f"\nTotal evaluation time for {target_voltage}V: {end_time - start_time:.2f} seconds")

                if plot_data:
                    plot_and_save_summary(
                        all_episode_data=plot_data, target_voltage=target_voltage,
                        tolerance=0.5, model_type=model_folder_name,
                        plot_save_base=PLOT_SAVE_BASE
                    )

            except FileNotFoundError as e:
                print(f"\n[ERROR] A required file was not found: {e}")
                print(f"Please ensure the folder '{training_run_folder}' exists and contains 'model_final.zip' and 'vec_normalize_final.pkl'.")
                break 
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
            finally:
                if env:
                    env.close()
                    print(f"\n--- Evaluation for {target_voltage}V complete. Environment closed. ---")

    print("\n" + "#"*80)
    print("# ALL EVALUATION RUNS COMPLETE")
    print("#"*80)
