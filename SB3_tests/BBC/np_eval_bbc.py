# jax_bbc_eval.py - Discover and evaluate trained buck-boost agents with advanced metrics

import argparse
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, SAC, TD3, DQN, PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

# We need the make_env function from the training script
from SB3_tests.BBC.np_bbc_train import make_env

# A dictionary mapping algorithm names (strings) to their class constructors
ALGO_MAP = {"a2c": A2C, "sac": SAC, "td3": TD3, "dqn": DQN, "ppo": PPO, "ddpg": DDPG}

def calculate_performance_metrics(voltages_arr, times_arr, target_voltage, tolerance):
    """
    Calculates detailed performance metrics from a single episode's voltage trajectory.
    Logic is adapted from the evaluate_buck.py script.
    """
    if len(times_arr) == 0:
        return {
            "stabilisation_time": np.nan, "steady_state_error": np.nan,
            "overshoot": np.nan, "undershoot": np.nan
        }

    # Find indices where the voltage is outside the tolerance band
    outside_tolerance_indices = np.where(np.abs(voltages_arr - target_voltage) > tolerance)[0]

    has_stabilized = False
    stabilisation_time = times_arr[-1]
    steady_state_error = np.nan

    if len(outside_tolerance_indices) == 0:
        # Stabilized from the very beginning
        first_stable_index = 0
        has_stabilized = True
        stabilisation_time = times_arr[0]
    else:
        # Find the last time it was unstable. Stabilization occurs at the next step.
        last_unstable_index = outside_tolerance_indices[-1]
        if last_unstable_index + 1 < len(times_arr):
            first_stable_index = last_unstable_index + 1
            stabilisation_time = times_arr[first_stable_index]
            has_stabilized = True

    if has_stabilized:
        steady_state_voltages = voltages_arr[first_stable_index:]
        steady_state_error = np.mean(steady_state_voltages - target_voltage)

    # Overshoot is the max voltage above target
    overshoot = max(0, np.max(voltages_arr) - target_voltage)

    # Undershoot is the max dip below target after the first time it crosses the target
    first_cross_indices = np.where(voltages_arr >= target_voltage)[0]
    if len(first_cross_indices) > 0:
        first_cross_index = first_cross_indices[0]
        voltages_after_cross = voltages_arr[first_cross_index:]
        undershoot = max(0, target_voltage - np.min(voltages_after_cross))
    else:
        undershoot = 0.0 # Never reached the target

    return {
        "stabilisation_time": stabilisation_time,
        "steady_state_error": steady_state_error,
        "overshoot": overshoot,
        "undershoot": undershoot
    }

def plot_and_save_summary(all_episode_data, target_voltage, tolerance, model_type, out_dir):
    """
    Plots voltage vs. time for all episodes, saves plots, and saves raw data.
    """
    # --- 1. Save Raw Data to CSV ---
    csv_path = os.path.join(out_dir, "all_episodes_trajectory.csv")
    all_data_to_save = []
    for i, (times, voltages, duties) in enumerate(all_episode_data):
        episode_col = np.full((len(times), 1), i + 1)
        episode_data = np.hstack((episode_col, times.reshape(-1, 1), voltages.reshape(-1, 1), duties.reshape(-1, 1)))
        all_data_to_save.append(episode_data)
    
    if all_data_to_save:
        final_data_array = np.vstack(all_data_to_save)
        np.savetxt(csv_path, final_data_array, delimiter=',', header='episode,time_s,voltage_v,duty_cycle', comments='')

    # --- 2. Plot Full View ---
    full_plot_path = os.path.join(out_dir, "response_plot_full.png")
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    for i, (times, voltages, _) in enumerate(all_episode_data):
        ax1.plot(times * 1000, voltages, label=f'Episode {i + 1}', alpha=0.8)
    ax1.axhline(y=target_voltage, color='r', linestyle='--', label='Target Voltage')
    ax1.axhspan(target_voltage - tolerance, target_voltage + tolerance, color='red', alpha=0.1, label=f'Tolerance (±{tolerance}V)')
    ax1.set_title(f'{model_type} Agent Performance - Full View', fontsize=16)
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Output Voltage (V)')
    ax1.legend(); ax1.grid(True)
    fig1.savefig(full_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # --- 3. Plot Zoomed View ---
    zoom_plot_path = os.path.join(out_dir, "response_plot_zoomed.png")
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    for i, (times, voltages, _) in enumerate(all_episode_data):
        ax2.plot(times * 1000, voltages, marker='.', linestyle='-', label=f'Episode {i + 1}', alpha=0.8)
    ax2.axhline(y=target_voltage, color='r', linestyle='--', label='Target Voltage')
    ax2.axhspan(target_voltage - tolerance, target_voltage + tolerance, color='red', alpha=0.1)
    ax2.set_title(f'{model_type} Agent Performance - Zoomed View', fontsize=16)
    ax2.set_xlabel('Time (ms)'); ax2.set_ylabel('Output Voltage (V)')
    ax2.set_ylim(target_voltage - (tolerance * 4), target_voltage + (tolerance * 4))
    ax2.legend(); ax2.grid(True)
    fig2.savefig(zoom_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"  - Saved detailed CSV and plots to: {out_dir}")


def evaluate_agent(
    model_path: str, stats_path: str, algo: str, out_dir: str, num_episodes: int,
    k_macro: int, dqn_bins: int, voltage_noise_std: float, tolerance: float
):
    """
    Loads and evaluates a single model, calculating and saving detailed metrics.
    """
    # Env setup
    env_fn = make_env(
        seed=int(time.time()), rank=0, k_macro=k_macro, algo_name=algo,
        dqn_bins=dqn_bins, voltage_noise_std=voltage_noise_std
    )
    env = DummyVecEnv([env_fn])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    
    # Extract env parameters
    target_voltage = env.get_attr("target_voltage")[0]
    dt_sim = env.get_attr("dt")[0]
    frame_skip = env.get_attr("frame_skip")[0]
    dt_step = dt_sim * frame_skip * k_macro

    model = ALGO_MAP[algo].load(model_path, env=env)
    print(f"  - Model, environment, and stats loaded successfully.")

    # Data collection lists
    all_metrics = []
    all_episode_data = []

    for i in tqdm(range(num_episodes), desc="  - Running Episodes"):
        obs = env.reset()
        done = False
        vC_trajectory, duty_trajectory = [], []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            
            info = infos[0]
            vC_trajectory.append(info.get("vC", np.nan))
            duty_trajectory.append(info.get("duty_cmd", np.nan))

        voltages_arr = np.array(vC_trajectory)
        duties_arr = np.array(duty_trajectory)
        times_arr = np.arange(len(voltages_arr)) * dt_step
        
        all_episode_data.append((times_arr, voltages_arr, duties_arr))
        
        # Calculate metrics for this episode
        metrics = calculate_performance_metrics(voltages_arr, times_arr, target_voltage, tolerance)
        metrics["reward"] = info.get("episode", {}).get("r", np.nan)
        metrics["length"] = info.get("episode", {}).get("l", np.nan)
        all_metrics.append(metrics)

    # Aggregate and summarize results
    run_name = os.path.basename(os.path.dirname(model_path))
    try:
        training_noise = float(run_name.split('noise_')[-1])
    except (ValueError, IndexError):
        training_noise = np.nan

    summary = {
        "run_name": run_name,
        "algo": algo, "episodes": num_episodes,
        "training_voltage_noise_std": training_noise,
        "evaluation_voltage_noise_std": voltage_noise_std,
        "mean_reward": np.nanmean([m["reward"] for m in all_metrics]),
        "std_reward": np.nanstd([m["reward"] for m in all_metrics]),
        "mean_stabilisation_time_s": np.nanmean([m["stabilisation_time"] for m in all_metrics]),
        "mean_steady_state_error_v": np.nanmean([m["steady_state_error"] for m in all_metrics]),
        "mean_overshoot_v": np.nanmean([m["overshoot"] for m in all_metrics]),
        "mean_undershoot_v": np.nanmean([m["undershoot"] for m in all_metrics]),
    }

    # Print summary to console
    print("\n  --- Evaluation Summary ---")
    print(f"  Mean Reward             : {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  Mean Stabilisation Time : {summary['mean_stabilisation_time_s']*1000:.2f} ms")
    print(f"  Mean Steady-State Error : {summary['mean_steady_state_error_v']*1000:.2f} mV")
    print(f"  Mean Overshoot          : {summary['mean_overshoot_v']*1000:.2f} mV")
    print(f"  Mean Undershoot         : {summary['mean_undershoot_v']*1000:.2f} mV")
    print("  --------------------------")

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    plot_and_save_summary(all_episode_data, target_voltage, tolerance, algo.upper(), out_dir)
    env.close()

def infer_algo_from_name(folder_name: str) -> str | None:
    for key in ALGO_MAP.keys():
        if folder_name.lower().startswith(key): return key
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover and evaluate trained buck-boost agents with advanced metrics.")
    parser.add_argument("--root", default="jax_models", help="Root directory containing trained model folders.")
    parser.add_argument("--algo", default="all", help=f"Algorithm to evaluate. Choices: {list(ALGO_MAP.keys())} or 'all'.")
    parser.add_argument("--name-contains", default=None, help="Only evaluate folders containing this substring (e.g., 'noise_0.010').")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes per model.")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Voltage tolerance (V) for stabilization metrics.")
    parser.add_argument("--macro-k", type=int, default=1, help="Macro-step value (k) used during training.")
    parser.add_argument("--dqn-bins", type=int, default=17, help="Number of duty bins for DQN models.")
    parser.add_argument("--eval-noise", type=float, nargs='+', default=[0.0], help="One or more evaluation-time noise std values to test against.")
    
    args = parser.parse_args()

    # --- 1. Discover Runs ---
    runs_to_evaluate = []
    print(f"🔍 Searching for models in '{args.root}'...")
    for folder_name in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, folder_name)
        if not os.path.isdir(run_dir): continue

        inferred_algo = infer_algo_from_name(folder_name)
        if not inferred_algo: continue
        if args.algo.lower() != "all" and inferred_algo != args.algo.lower(): continue
        if args.name_contains and args.name_contains not in folder_name: continue
            
        model_path = os.path.join(run_dir, "best_model.zip")
        stats_path = os.path.join(run_dir, f"{inferred_algo}_vec_normalize_final.pkl")
        
        if os.path.exists(model_path) and os.path.exists(stats_path):
            runs_to_evaluate.append({
                "run_name": folder_name, "model_path": model_path,
                "stats_path": stats_path, "algo": inferred_algo
            })
    
    if not runs_to_evaluate:
        print("\nNo valid models found matching the criteria."); exit()
    print(f"✅ Found {len(runs_to_evaluate)} models to evaluate.")

    # --- 2. Run Evaluation Loop ---
    for eval_nl in args.eval_noise:
        print(f"\n{'='*20} Starting Evaluation Pass: Environment Noise = {eval_nl:.3f} {'='*20}")
        for run in runs_to_evaluate:
            run_name = run['run_name']
            algo_key = run['algo'].upper()
            
            # Create the hierarchical output directory
            eval_condition_name = f"{run['algo'].lower()}_eval_noise_{eval_nl:.3f}"
            output_directory = os.path.join("eval_runs", algo_key, eval_condition_name, run_name)
            
            print(f"\n▶️ Evaluating run: {run_name}")
            
            evaluate_agent(
                model_path=run['model_path'], 
                stats_path=run['stats_path'],
                algo=run['algo'], 
                out_dir=output_directory,
                num_episodes=args.episodes, 
                k_macro=args.macro_k,
                dqn_bins=args.dqn_bins, 
                voltage_noise_std=eval_nl, # Pass the evaluation noise
                tolerance=args.tolerance
            )

    print("\n🎉 All evaluations complete.")
