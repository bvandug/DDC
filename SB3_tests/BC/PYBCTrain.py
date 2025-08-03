import os
import torch
import numpy as np
from datetime import datetime
from torch import nn
import json
import matplotlib.pyplot as plt
from collections import deque
import time

from stable_baselines3 import SAC, A2C, TD3, PPO, DDPG
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

# Using the Python-based environment for this training script
from PYBCEnv import BuckConverterEnv

def process_manual_hyperparams(raw_params, algo_name):
    """
    Processes a dictionary of manually set hyperparameters into the format
    required by the Stable Baselines3 model.
    """
    params = raw_params.copy()
    
    activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
    activation_fn_name = params.pop("activation_fn")
    activation_fn = activation_map.get(activation_fn_name, nn.ReLU)
    
    n_layers = params.pop("n_layers")
    layer_size = params.pop("layer_size")
    net_arch_list = [layer_size] * n_layers

    policy_kwargs = {"activation_fn": activation_fn}
    
    algo = algo_name.lower()
    if algo in ["a2c", "ppo"]:
        policy_kwargs["net_arch"] = dict(pi=net_arch_list, vf=net_arch_list)
    elif algo == "sac":
        policy_kwargs["net_arch"] = dict(pi=net_arch_list, qf=net_arch_list)
    elif algo in ["td3", "ddpg"]:
        policy_kwargs["net_arch"] = net_arch_list

    params["policy_kwargs"] = policy_kwargs

    if algo in ["td3", "ddpg"] and "action_noise_sigma" in params:
        action_noise_sigma = params.pop("action_noise_sigma")
        params["action_noise"] = NormalActionNoise(mean=np.zeros(1), sigma=action_noise_sigma * np.ones(1))
        
    return params

class TensorBoardCallback(BaseCallback):
    """
    A custom callback that logs raw and mean episode rewards to TensorBoard.
    """
    def __init__(self, mean_reward_window: int = 50, verbose: int = 0):
        super(TensorBoardCallback, self).__init__(verbose)
        self.current_episode_rewards = None
        self.all_episode_rewards = []
        self.mean_reward_buffer = deque(maxlen=mean_reward_window)

    def _on_training_start(self) -> None:
        self.current_episode_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        self.current_episode_rewards += self.locals['rewards']
        for i, done in enumerate(self.locals['dones']):
            if done:
                episode_reward = self.current_episode_rewards[i]
                self.mean_reward_buffer.append(episode_reward)
                self.logger.record("episode/reward", episode_reward)
                self.logger.record("episode/mean_reward", np.mean(self.mean_reward_buffer))
                self.logger.dump(step=self.num_timesteps)
                self.all_episode_rewards.append(episode_reward)
                self.current_episode_rewards[i] = 0
        return True

# <<< NEW: Evaluation and plotting functions from your evaluation script >>>
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
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            unnormalized_obs = env.get_original_obs()
            current_voltage = unnormalized_obs[0][0]
            current_time = env.get_attr('total_sim_time')[0]

            obs, reward, dones, info = env.step(action)
            done = dones[0]

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

if __name__ == "__main__":
    drive_base_path = "./DDC_RL_A2C"

    TOP_3_A2C_TRIALS = {
        #"Trial27": {
        #    'n_layers': 2, 'layer_size': 81, 'activation_fn': 'tanh',
        #    'learning_rate': 0.0030751064383758345, 'gamma': 0.9126672103243557,
        #    'ent_coef': 0.0002695884022707838, 'vf_coef': 0.289501136567359,
        #    'max_grad_norm': 2.653017568726791, 'n_steps': 495, 'gae_lambda': 0.9417019979198915
        #},
        "Trial7": {
            'n_layers': 1, 'layer_size': 115, 'activation_fn': 'tanh',
            'learning_rate': 0.00324216113440775, 'gamma': 0.9299591477007767,
            'ent_coef': 1.8756945609332168e-05, 'vf_coef': 0.4490098411745866,
            'max_grad_norm': 3.0314754648510203, 'n_steps': 454, 'gae_lambda': 0.9324082185363809
        },
        #"Trial38": {
        #    'n_layers': 2, 'layer_size': 210, 'activation_fn': 'tanh',
        #    'learning_rate': 0.0032931757757879198, 'gamma': 0.9014530385908368,
        #    'ent_coef': 0.003391193403035329, 'vf_coef': 0.41467978817945633,
        #    'max_grad_norm': 3.5736739089331557, 'n_steps': 276, 'gae_lambda': 0.9075555617480485
        #}
    }

    for trial_name, raw_hyperparams in TOP_3_A2C_TRIALS.items():
        print("\n" + "#"*80)
        print(f"# STARTING TRAINING & EVALUATION FOR A2C - {trial_name}")
        print("#"*80 + "\n")

        MODEL_TO_TRAIN = 'A2C'
        SEED = 42
        total_timesteps = 400000

        hyperparams = process_manual_hyperparams(raw_hyperparams, MODEL_TO_TRAIN)
        print(f"Successfully loaded manual hyperparameters for {trial_name}.")

        set_random_seed(SEED, using_cuda=False)

        folder_name = f"{MODEL_TO_TRAIN}_{trial_name}_Seed{SEED}"
        base_save_path = os.path.join(drive_base_path, folder_name)
        os.makedirs(base_save_path, exist_ok=True)
        print(f"All files will be saved to: {base_save_path}")
        tensorboard_log_path = os.path.join(base_save_path, "tensorboard_log/")

        env_params = {
            'use_randomized_goal': True,
            'target_voltage_min': 27.0,
            'target_voltage_max': 33.0,
        }
        env_fn = lambda: BuckConverterEnv(**env_params)
        env = DummyVecEnv([env_fn]); env.seed(SEED)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        model_class = globals()[MODEL_TO_TRAIN]
        model = model_class("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_path, seed=SEED, **hyperparams)

        tensorboard_callback = TensorBoardCallback()
        checkpoint_callback = CheckpointCallback(
          save_freq=50000, save_path=base_save_path, name_prefix="model_checkpoint")
          
        print(f"\n--- Starting Training for {MODEL_TO_TRAIN} ({total_timesteps} Timesteps) ---")
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=[tensorboard_callback, checkpoint_callback])

        model.save(os.path.join(base_save_path, "model_final"))
        env.save(os.path.join(base_save_path, "vec_normalize_final.pkl"))
        print(f"\n--- Final {MODEL_TO_TRAIN} Model for {trial_name} Saved ---")
        env.close()

        # --- MODIFIED: Integrated Evaluation Section ---
        print("\n" + "="*80)
        print(f"# STARTING EVALUATION FOR {trial_name}")
        print("="*80 + "\n")

        EVALUATION_SAVE_DIR = "./evaluation_results"
        GENERALIZATION_VOLTAGES = [25, 27.5, 29.0, 30.0, 31.5, 32.5, 35]
        N_EVAL_EPISODES = 5
        MAX_EPISODE_STEPS = 600

        MODEL_LOAD_PATH = os.path.join(base_save_path, "model_final.zip")
        STATS_LOAD_PATH = os.path.join(base_save_path, "vec_normalize_final.pkl")

        for target_voltage in GENERALIZATION_VOLTAGES:
            print("\n" + "-"*80)
            print(f"--- Evaluating {trial_name} for Target Voltage: {target_voltage}V ---")
            print("-" * 80 + "\n")

            plot_save_folder = os.path.join(EVALUATION_SAVE_DIR, folder_name)
            os.makedirs(plot_save_folder, exist_ok=True)
            PLOT_SAVE_BASE = os.path.join(plot_save_folder, f"eval_at_{target_voltage:.1f}V")

            eval_env = None
            try:
                eval_env_fn = lambda: BuckConverterEnv(
                    use_randomized_goal=False,
                    fixed_goal_voltage=target_voltage,
                    max_episode_steps=MAX_EPISODE_STEPS
                )
                
                eval_env = DummyVecEnv([eval_env_fn])
                eval_env = VecNormalize.load(STATS_LOAD_PATH, eval_env)
                eval_env.training = False
                eval_env.norm_reward = False
                
                eval_model = model_class.load(MODEL_LOAD_PATH, env=eval_env)
                
                plot_data = run_evaluation(
                    model=eval_model, env=eval_env, n_episodes=N_EVAL_EPISODES,
                    target_voltage=target_voltage, tolerance=0.5
                )

                if plot_data:
                    plot_and_save_summary(
                        all_episode_data=plot_data, target_voltage=target_voltage,
                        tolerance=0.5, model_type=f"{MODEL_TO_TRAIN}_{trial_name}",
                        plot_save_base=PLOT_SAVE_BASE
                    )
            finally:
                if eval_env:
                    eval_env.close()

        print(f"\n--- Finished training and evaluation for {trial_name} ---")

    print("\n" + "#"*80)
    print("# ALL TRAINING RUNS COMPLETE")
    print("#"*80)
