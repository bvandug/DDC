import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG, PPO, SAC, TD3, DQN, A2C
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from torch import nn
import random
import torch
from tqdm import tqdm
import json
import gymnasium as gym
from gymnasium import spaces

# --- Import The Python Environment ---
# NOTE: This script assumes your custom 'PYBCEnv' environment is defined in 'PYBCEnv.py'
from PYBCEnv import BuckConverterEnv as BCPyEnv


# --- Helper function for plotting (Unchanged) ---
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

def run_evaluation(model, stats_path, n_episodes=5, target_voltage=30.0, tolerance=0.5, max_episode_steps=2000):
    """Evaluates a trained model on the Python environment."""
    all_rewards, stabilisation_times, steady_state_errors, overshoots = [], [], [], []
    all_episode_plot_data = []

    eval_env = None
    try:
        is_dqn = isinstance(model, DQN)
        
        def make_eval_env():
            env = BCPyEnv(use_randomized_goal=False, fixed_goal_voltage=target_voltage, max_episode_steps=max_episode_steps, voltage_noise_std=0.0)
            if is_dqn:
                env = DiscretizeActionWrapper(env, n_bins=17)
            return env

        eval_env = DummyVecEnv([make_eval_env])
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        for ep in range(n_episodes):
            obs = eval_env.reset()
            ep_reward = 0.0
            episode_voltages, episode_times = [], []
            
            print(f"\n--- Running Evaluation Episode {ep + 1}/{n_episodes} at {target_voltage}V ---")
    
            while True:
                action, _ = model.predict(obs, deterministic=True)
                
                unnormalized_obs = eval_env.get_original_obs()
                current_voltage = unnormalized_obs[0][0]
                current_time = eval_env.get_attr('total_sim_time')[0]
                
                obs, reward, dones, info = eval_env.step(action)
                done = dones[0]

                ep_reward += reward[0]
                episode_voltages.append(current_voltage)
                episode_times.append(current_time)

                if done:
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


class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_bins=17):
        super().__init__(env)
        self.n_bins = n_bins
        self.action_space = spaces.Discrete(self.n_bins)
        self.continuous_actions = np.linspace(
            self.env.action_space.low[0],
            self.env.action_space.high[0],
            self.n_bins
        )
    def action(self, action):
        continuous_action = self.continuous_actions[action]
        return np.array([continuous_action], dtype=np.float32)

def run_experiment(algo_name, hyperparams, seed, main_save_path, total_timesteps, noise_std):
    rank = hyperparams["rank"]
    run_save_path = os.path.join(main_save_path, f"{algo_name}_Rank_{rank}_Seed_{seed}_Noise_{noise_std}")
    tb_log_path = os.path.join(run_save_path, "tensorboard_logs")
    eval_save_path = os.path.join(run_save_path, "generalization_tests")
    os.makedirs(tb_log_path, exist_ok=True)
    os.makedirs(eval_save_path, exist_ok=True)
    
    model_save_path = os.path.join(run_save_path, "final_model.zip")
    stats_save_path = os.path.join(run_save_path, "vec_normalize.pkl")
    params_save_path = os.path.join(run_save_path, "hyperparameters.json")
    
    with open(params_save_path, "w") as f:
        json.dump(hyperparams, f, indent=4)

    print("\n" + "#"*80)
    print(f"# --- STARTING TRAINING: {algo_name}, Rank {rank}, Seed {seed}, Noise Level: {noise_std} ---")
    print(f"Training for {total_timesteps} timesteps.")
    print("#"*80 + "\n")

    train_env = None
    try:
        set_random_seed(seed)
        
        def make_env(noise_level):
            env = BCPyEnv(use_randomized_goal=True, target_voltage_min=28.5, target_voltage_max=31.5, voltage_noise_std=noise_level)
            if algo_name.lower() == "dqn":
                env = DiscretizeActionWrapper(env, n_bins=17)
            return Monitor(env)

        train_env = DummyVecEnv([lambda: make_env(noise_std)])
        train_env.seed(seed)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}
        model_params = hyperparams.copy()
        policy_kwargs = {}
        if "n_layers" in model_params and "layer_size" in model_params and "activation_fn" in model_params:
            n_layers = model_params.pop("n_layers")
            layer_size = model_params.pop("layer_size")
            activation_fn_name = model_params.pop("activation_fn")
            activation_fn = activation_map.get(activation_fn_name, nn.ReLU)
            net_arch = [layer_size] * n_layers
            if algo_name.lower() in ["a2c", "ppo"]:
                policy_kwargs["net_arch"] = dict(pi=net_arch, vf=net_arch)
            else:
                policy_kwargs["net_arch"] = net_arch
            policy_kwargs["activation_fn"] = activation_fn

        model_params.pop("rank", None)
        
        if algo_name.lower() in ["ddpg", "td3"]:
            action_noise_sigma = model_params.pop("action_noise_sigma")
            n_actions = train_env.action_space.shape[-1]
            model_params["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=action_noise_sigma * np.ones(n_actions))
        if "clip_range" in model_params: model_params["clip_range"] = float(model_params["clip_range"])
        if "ent_coef" in model_params and model_params["ent_coef"] == "auto": pass
        elif "ent_coef" in model_params: model_params["ent_coef"] = float(model_params["ent_coef"])
        if "batch_size" in model_params: model_params["batch_size"] = int(model_params["batch_size"])
        if "buffer_size" in model_params: model_params["buffer_size"] = int(model_params["buffer_size"])
        if "learning_starts" in model_params: model_params["learning_starts"] = int(model_params["learning_starts"])
        
        # --- THIS LINE HAS BEEN REMOVED TO FIX THE A2C/PPO CRASH ---
        # model_params.pop("n_steps", None)
        
        model_class = globals()[algo_name]
        model = model_class("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log=tb_log_path, seed=seed, **model_params)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        
        print(f"\n--- Training complete. Saving final model to: {model_save_path} ---")
        model.save(model_save_path)
        train_env.save(stats_save_path)
        print("--- Model and stats saved successfully. ---")

    finally:
        if train_env:
            train_env.close()
            print("\nPython training environment closed.")

    print("\n" + "#"*80)
    print(f"# --- STARTING EVALUATION: {algo_name}, Rank {rank}, Seed {seed}, Noise Level: {noise_std} ---")
    print(f"Evaluating on Python environment (with zero noise).")
    print("#"*80 + "\n")

    model_to_eval = model_class.load(model_save_path)
    
    for voltage in EVALUATION_VOLTAGES:
        print("\n" + "="*80)
        print(f"--- EVALUATING AT TARGET VOLTAGE: {voltage}V ---")
        print("="*80)
        
        plot_data = run_evaluation(
            model=model_to_eval,
            stats_path=stats_save_path,
            n_episodes=1,
            target_voltage=voltage,
            max_episode_steps=EVAL_EPISODE_STEPS
        )
        
        if plot_data:
            plot_save_base = os.path.join(eval_save_path, f"evaluation_at_{voltage:.1f}V")
            plot_and_save_summary(plot_data, voltage, 0.5, algo_name, plot_save_base)


if __name__ == '__main__':
    TOTAL_TRAINING_TIMESTEPS = 400000
    OFF_POLICY_TIMESTEPS = 200000
    DQN_TIMESTEPS = 300000
    EVALUATION_VOLTAGES = [25, 27.5, 29.0, 30.0, 31.5, 32.5, 35]
    EVAL_EPISODE_STEPS = 2000
    
    NOISE_LEVELS = [0, 0.001, 0.01, 0.1]

    hyperparameter_sets = {
        "A2C": [
            {
              'rank': 1,
              'learning_rate': 0.002517700894695954,
              'gamma': 0.9852317519921416,
              'n_steps': 1024,
              'ent_coef': 0.00019716244332231868,
              'vf_coef': 0.3800239918811925,
              'max_grad_norm': 1.7018760638184602,
              'gae_lambda': 0.9416240334404364,
              'normalize_advantage': True,
              'n_layers': 2,
              'layer_size': 156,
              'activation_fn': 'relu'
          },
          {'rank': 3, 'n_layers': 1, 'layer_size': 100, 'activation_fn': 'relu', 'learning_rate': 0.004627189994055962, 'gamma': 0.9464734604761008, 'ent_coef': 1.5005587871905613e-05, 'vf_coef': 0.6933384386464978, 'max_grad_norm': 1.1798883043031796, 'n_steps': 112, 'gae_lambda': 0.9786634896219476}
        ],
        "PPO": [
            {'rank': 2, 'n_layers': 3, 'layer_size': 148, 'activation_fn': 'tanh', 'learning_rate': 0.0001830572579280076, 'gamma': 0.9763544512197304, 'ent_coef': 2.964544717414107e-08, 'vf_coef': 0.6977323388095654, 'max_grad_norm': 3.2852985886011443, 'n_steps': 2048, 'batch_size': 128, 'n_epochs': 18, 'clip_range': 0.2077936161045162, 'gae_lambda': 0.9310857040746142}
        ],
        "SAC": [
            {'rank': 1, 'n_layers': 3, 'layer_size': 105, 'activation_fn': 'leaky_relu', 'learning_rate': 0.0004988313346799755, 'buffer_size': 150306, 'batch_size': 256, 'tau': 0.00661542279269948, 'gamma': 0.932069775803889, 'ent_coef': 'auto'}
        ],
        "DDPG": [
            {'rank': 1, 'n_layers': 3, 'layer_size': 132, 'activation_fn': 'tanh', 'action_noise_sigma': 0.1892751990632762, 'learning_rate': 3.984749428410908e-05, 'buffer_size': 153741, 'batch_size': 128, 'gamma': 0.9683432382369432, 'tau': 0.01904684256059171, 'train_freq': 32, 'gradient_steps': 16}
        ],
        "TD3": [
            {'rank': 1, 'n_layers': 3, 'layer_size': 256, 'activation_fn': 'relu', 'action_noise_sigma': 0.28111990104794127, 'learning_rate': 0.0003252676129966604, 'buffer_size': 101461, 'batch_size': 326, 'tau': 0.016287309398475194, 'gamma': 0.9118527469761414, 'policy_delay': 4, 'target_policy_noise': 0.12838497740816027, 'target_noise_clip': 0.5650877741516667, 'gradient_steps': 8, 'train_freq': 8}
        ],
        "DQN": [
            {'rank': 1, 'n_layers': 2, 'layer_size': 309, 'activation_fn': 'tanh', 'learning_rate': 0.00025853041708717625, 'buffer_size': 121708, 'learning_starts': 7202, 'batch_size': 128, 'gamma': 0.9159261928463776, 'exploration_fraction': 0.13204826185966512, 'exploration_final_eps': 0.10800385063818738, 'train_freq': 1, 'target_update_interval': 510}
        ]
    }
    
    seed = 42
    for noise_level in NOISE_LEVELS:
        for algo_name, top_params_list in hyperparameter_sets.items():
            main_test_folder = os.path.join("./final_models_noise_new", algo_name)
            os.makedirs(main_test_folder, exist_ok=True)
            
            training_timesteps = TOTAL_TRAINING_TIMESTEPS if algo_name in ["A2C", "PPO"] else DQN_TIMESTEPS if algo_name == "DQN" else OFF_POLICY_TIMESTEPS
            
            for params_set in top_params_list:
                run_experiment(
                    algo_name=algo_name,
                    hyperparams=params_set, 
                    seed=seed,
                    main_save_path=main_test_folder,
                    total_timesteps=training_timesteps,
                    noise_std=noise_level
                )

    print("\n\n--- ALL TRAINING AND EVALUATION RUNS COMPLETE ---")