import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from torch import nn
import random
import torch
from tqdm import tqdm

# --- Import The Python Environment ---
# This environment will be used for both training and evaluation.
from PYBCEnv import BuckConverterEnv as BCPyEnv 

# --- Import for Google Drive Mounting ---
try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# --- Utility and Plotting Functions ---

def plot_and_save_summary(all_episode_data, target_voltage, tolerance, model_type, plot_save_base):
    """Plots the voltage vs. time for all evaluation episodes."""
    full_view_path = f"{plot_save_base}_{target_voltage}_{model_type}_full_view.png"
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
        ax1.set_ylim(bottom=min(target_voltage - 5, min_volt - 1), top=max(target_voltage + 5, max_volt + 1))
    fig1.savefig(full_view_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("--- Full view plot saved successfully! ---")

def run_evaluation(model_path, stats_path, n_episodes=5, target_voltage=30.0, tolerance=0.5, max_episode_steps=2000):
    """Evaluates a trained model on the Python environment."""
    all_rewards, stabilisation_times, steady_state_errors, overshoots = [], [], [], []
    all_episode_plot_data = []

    eval_env = None
    try:
        env_fn = lambda: BCPyEnv(use_randomized_goal=False, fixed_goal_voltage=target_voltage, max_episode_steps=max_episode_steps)
        eval_env = DummyVecEnv([env_fn])
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        loaded_model = DDPG.load(model_path, env=eval_env)

        for ep in range(n_episodes):
            obs = eval_env.reset()
            ep_reward = 0.0
            episode_voltages, episode_times = [], []
            
            print(f"\n--- Running Evaluation Episode {ep + 1}/{n_episodes} ---")
            
            while True:
                action, _ = loaded_model.predict(obs, deterministic=True)
                
                unnormalized_obs = eval_env.get_original_obs()
                current_voltage = unnormalized_obs[0][0]
                current_time = eval_env.get_attr('total_sim_time')[0]
                
                obs, reward, dones, info = eval_env.step(action)
                
                ep_reward += reward[0]
                episode_voltages.append(current_voltage)
                episode_times.append(current_time)

                if dones[0]:
                  break

            times_arr = np.array(episode_times)
            voltages_arr = np.array(episode_voltages)

            if len(times_arr) == 0: continue
            
            all_episode_plot_data.append((times_arr, voltages_arr))

            has_stabilized = False
            stabilisation_time = times_arr[-1]
            outside_tolerance_indices = np.where(np.abs(voltages_arr - target_voltage) > tolerance)[0]
            if len(outside_tolerance_indices) == 0:
                has_stabilized = True
                stabilisation_time = times_arr[0] if len(times_arr) > 0 else 0
            else:
                last_unstable_index = outside_tolerance_indices[-1]
                if last_unstable_index + 1 < len(times_arr):
                    has_stabilized = True
                    stabilisation_time = times_arr[last_unstable_index + 1]
            
            steady_error = np.mean(voltages_arr[np.where(times_arr >= stabilisation_time)] - target_voltage) if has_stabilized else np.nan
            overshoot = max(0, np.max(voltages_arr) - target_voltage) if len(voltages_arr) > 0 else 0
            
            all_rewards.append(ep_reward)
            stabilisation_times.append(stabilisation_time)
            overshoots.append(overshoot)
            steady_state_errors.append(steady_error)

    finally:
        if eval_env:
            eval_env.close()

    print("\n--- Overall Evaluation Summary ---")
    print("-" * 40)
    print(f"Mean Reward             : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean Stabilisation Time : {np.mean(stabilisation_times) * 1000:.2f} ms")
    with np.errstate(invalid='ignore'):
        print(f"Mean Steady-State Error : {np.nanmean(steady_state_errors):.4f} V")
    print(f"Mean Overshoot          : {np.mean(overshoots):.4f} V")
    print("-" * 40)
    
    return all_episode_plot_data

def run_experiment(hyperparams, seed, main_save_path):
    """
    Trains and evaluates a single DDPG model configuration with a specific seed.
    """
    rank = hyperparams["rank"]
    
    run_save_path = os.path.join(main_save_path, f"Rank_{rank}_Seed_{seed}")
    tb_log_path = os.path.join(run_save_path, "tensorboard_logs")
    eval_save_path = os.path.join(run_save_path, "generalization_tests")
    os.makedirs(tb_log_path, exist_ok=True)
    os.makedirs(eval_save_path, exist_ok=True)
    
    model_save_path = os.path.join(run_save_path, "final_model.zip")
    stats_save_path = os.path.join(run_save_path, "vec_normalize.pkl")

    # ======================================================================
    # 1. TRAINING PHASE (using Python Environment)
    # ======================================================================
    print("\n" + "#"*80)
    print(f"# --- STARTING TRAINING: Rank {rank}, Seed {seed} ---")
    print(f"Training for {TOTAL_TRAINING_TIMESTEPS} timesteps.")
    print("#"*80 + "\n")

    train_env = None
    try:
        set_random_seed(seed)
        env_fn = lambda: Monitor(BCPyEnv(use_randomized_goal=True, target_voltage_min=28.5, target_voltage_max=31.5))
        train_env = DummyVecEnv([env_fn])
        train_env.seed(seed)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        policy_kwargs = {
            "net_arch": [hyperparams["layer_size"]] * hyperparams["n_layers"],
            "activation_fn": {"tanh": nn.Tanh, "relu": nn.ReLU}[hyperparams["activation_fn"]]
        }
        
        n_actions = train_env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), 
            sigma=hyperparams["action_noise_sigma"] * np.ones(n_actions)
        )

        model_params = hyperparams.copy()
        for key in ["rank", "n_layers", "layer_size", "activation_fn", "action_noise_sigma"]:
            del model_params[key]

        model = DDPG(
            "MlpPolicy", 
            train_env, 
            policy_kwargs=policy_kwargs, 
            action_noise=action_noise,
            verbose=1, 
            tensorboard_log=tb_log_path, 
            seed=seed, 
            **model_params
        )
        model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS, progress_bar=True)
        
        print(f"\n--- Training complete. Saving final model to: {run_save_path} ---")
        model.save(model_save_path)
        train_env.save(stats_save_path)
        print("--- Model and stats saved successfully. ---")

    finally:
        if train_env:
            train_env.close()
            print("\nPython training environment closed.")

    # ======================================================================
    # 2. EVALUATION PHASE (using Python Environment)
    # ======================================================================
    print("\n" + "#"*80)
    print(f"# --- STARTING EVALUATION: Rank {rank}, Seed {seed} ---")
    print(f"Evaluating on Python environment.")
    print("#"*80 + "\n")
    
    for voltage in EVALUATION_VOLTAGES:
        print("\n" + "="*80)
        print(f"--- EVALUATING AT TARGET VOLTAGE: {voltage}V ---")
        print("="*80)
        
        plot_data = run_evaluation(
            model_path=model_save_path,
            stats_path=stats_save_path,
            n_episodes=1,
            target_voltage=voltage,
            max_episode_steps=EVAL_EPISODE_STEPS
        )
        
        if plot_data:
            plot_save_base = os.path.join(eval_save_path, f"evaluation_at_{voltage:.1f}V")
            plot_and_save_summary(plot_data, voltage, 0.5, "DDPG", plot_save_base)


# --- Main Script ---
if __name__ == '__main__':
    # --- Define Base Path ---
    if IS_COLAB:
        print("Running in Google Colab. Assuming Drive is already mounted.")
        gdrive_base_path = "/content/drive/MyDrive/DDC"
    else:
        gdrive_base_path = "./DDC"
        
    main_test_folder = os.path.join(gdrive_base_path, "DDPG_Final_Params")
    os.makedirs(main_test_folder, exist_ok=True)
    print(f"All results will be saved in: {main_test_folder}")

    # --- Configuration ---
    TOTAL_TRAINING_TIMESTEPS = 200000
    EVALUATION_VOLTAGES = [27.5, 29.0, 30.0, 31.5, 32.5]
    EVAL_EPISODE_STEPS = 2000
    
    # --- The top 2 DDPG hyperparameter sets from your Optuna study ---
    experiments_to_run = [
        {
            "seed": 663996,
            "params": {'rank': 2, 'n_layers': 3, 'layer_size': 327, 'activation_fn': 'tanh', 'learning_rate': 0.0002061163011738982, 'buffer_size': 237949, 'batch_size': 128, 'gamma': 0.9046223214626081, 'tau': 0.012670170106082904, 'train_freq': 8, 'gradient_steps': 8, 'action_noise_sigma': 0.057700908370714184}
        }
    ]
    
    # --- Main Loop for Experiments ---
    for experiment in experiments_to_run:
        run_experiment(
            hyperparams=experiment["params"], 
            seed=experiment["seed"],
            main_save_path=main_test_folder
        )

    print("\n\n--- ALL TRAINING AND EVALUATION RUNS COMPLETE ---")
