import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
import gym 
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from BCSimulinkEnv import BCSimulinkEnv
from stable_baselines3 import SAC, TD3, A2C, DDPG, DQN, PPO
from tqdm import tqdm

# --- The DiscretizeActionWrapper class ---
class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    A wrapper to discretize a continuous action space for DQN.
    """
    def __init__(self, env, n_bins=17):
        super().__init__(env)
        self.n_bins = n_bins
        self.action_space = gym.spaces.Discrete(self.n_bins)
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
        continuous_action = self.continuous_actions[action]
        return np.array([continuous_action], dtype=np.float32)


def plot_combined_summary(all_models_data, base_save_path):
    """
    Generates a summary plot comparing all models at a single, representative voltage.
    """
    print("\n--- Generating Combined Summary Plot ---")

    comparison_voltage = 30.0
    plot_path_model_comp = os.path.join(base_save_path, f"all_models_comparison_{comparison_voltage}V.png")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_models_data)))

    for i, (model_name, model_data) in enumerate(all_models_data.items()):
        for times, voltages, goal in model_data:
            if goal == comparison_voltage:
                # --- FIXED: Convert list to numpy array before multiplying ---
                times_ms = np.array(times) * 1000
                ax1.plot(times_ms, voltages, label=model_name, color=colors[i], alpha=0.8)
                # ---
                break 

    ax1.axhline(y=comparison_voltage, color='r', linestyle='--', linewidth=2, label=f'Target Voltage ({comparison_voltage}V)')
    ax1.set_title(f'All Models Performance Comparison at {comparison_voltage}V', fontsize=16, weight='bold')
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Output Voltage (V)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim(bottom=comparison_voltage - 5, top=comparison_voltage + 5)
    
    fig1.savefig(plot_path_model_comp, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"--- Model comparison plot saved to {plot_path_model_comp} ---")


def run_evaluation(model, env, n_episodes=1, target_voltage=30.0, tolerance=0.5):
    """
    Evaluates a trained model and returns plot data and detailed metrics for a single episode.
    """
    all_episode_plot_data = []
    
    env.env_method('set_goal_voltage', target_voltage)
    
    pbar = tqdm(desc=f"Evaluating {target_voltage}V", unit=" step")
    is_first_step_of_eval = True

    # Since n_episodes is 1, this loop runs once
    obs = env.reset()
    done = False
    ep_reward = 0.0
    episode_voltages, episode_times = [], []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        
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
        
        if dones[0]:
            done = True

    times_arr = np.array(episode_times[:-1])
    voltages_arr = np.array(episode_voltages[:-1])

    if len(times_arr) > 0:
        all_episode_plot_data.append((times_arr.tolist(), voltages_arr.tolist(), target_voltage))

    # --- Full Metric Calculation for the single episode ---
    has_stabilized = False
    stabilisation_time = times_arr[-1] if len(times_arr) > 0 else 0.0
    stable_time_for = 0.0
    
    outside_tolerance_indices = np.where(np.abs(voltages_arr - target_voltage) > tolerance)[0]
    if len(outside_tolerance_indices) == 0 and len(times_arr) > 0:
        first_stable_index = 0
        has_stabilized = True
        stabilisation_time = times_arr[0]
    elif len(outside_tolerance_indices) > 0:
        last_unstable_index = outside_tolerance_indices[-1]
        if last_unstable_index + 1 < len(times_arr):
            first_stable_index = last_unstable_index + 1
            has_stabilized = True
            stabilisation_time = times_arr[first_stable_index]

    if has_stabilized:
        steady_state_voltages = voltages_arr[first_stable_index:]
        steady_error = np.mean(steady_state_voltages - target_voltage)
        stable_time_for = times_arr[-1] - stabilisation_time
    else:
        steady_error = np.nan

    overshoot = max(0, np.max(voltages_arr) - target_voltage) if len(voltages_arr) > 0 else 0.0

    pbar.close()
    
    metrics = {
        "total_reward": float(ep_reward),
        "stabilisation_time_ms": float(stabilisation_time) * 1000 if has_stabilized else None,
        "stabilization_duration_ms": float(stable_time_for) * 1000 if has_stabilized else 0.0,
        "steady_state_error_V": float(steady_error) if not np.isnan(steady_error) else None,
        "overshoot_V": float(overshoot)
    }
    return all_episode_plot_data, metrics


if __name__ == '__main__':
    ALL_MODELS = {
        "A2C": A2C, "PPO": PPO, "SAC": SAC, 
        "TD3": TD3, "DDPG": DDPG, "DQN": DQN
    }
    EVALUATION_VOLTAGES = [30.0]
    N_EVAL_EPISODES = 1
    EVAL_EPISODE_TIME = 0.1

    all_models_evaluation_data = {}
    all_models_metrics_data = {}

    for model_name, model_class in ALL_MODELS.items():
        print("\n" + "="*80)
        print(f"STARTING EVALUATION FOR: {model_name}")
        print("="*80)

        MODEL_SAVE_PATH = f"final_model_{model_name}.zip"
        STATS_PATH = f"vec_normalize_{model_name}.pkl"
        
        for voltage in EVALUATION_VOLTAGES:
            env = None
            try:
                if model_name == 'DQN':
                    env_fn = lambda: DiscretizeActionWrapper(BCSimulinkEnv(model_name="bcSim", enable_plotting=False, fixed_goal_voltage=voltage, max_episode_time=EVAL_EPISODE_TIME))
                else:
                    env_fn = lambda: BCSimulinkEnv(model_name="bcSim", enable_plotting=False, fixed_goal_voltage=voltage, max_episode_time=EVAL_EPISODE_TIME)
                
                env = DummyVecEnv([env_fn])
                env = VecNormalize.load(STATS_PATH, env)
                env.training = False
                env.norm_reward = False
                
                model = model_class.load(MODEL_SAVE_PATH, env=env)
                
                plot_data, metrics_data = run_evaluation(
                    model=model, 
                    env=env, 
                    n_episodes=N_EVAL_EPISODES, 
                    target_voltage=voltage, 
                    tolerance=0.5
                )
                
                if plot_data:
                    all_models_evaluation_data[model_name] = plot_data
                if metrics_data:
                    all_models_metrics_data[model_name] = metrics_data
                    
                    print("\n" + "-"*40)
                    print(f"--- Final Metrics for {model_name} at {voltage}V ---")
                    for key, value in metrics_data.items():
                        if value is None:
                            print(f"  {key.replace('_', ' ').title():<25}: Not Achieved")
                        else:
                            print(f"  {key.replace('_', ' ').title():<25}: {value:.4f}")
                    print("-" * 40)


            except FileNotFoundError:
                print(f"\n[SKIPPING] Model file not found for {model_name}: {MODEL_SAVE_PATH}")
                break 
            except Exception as e:
                print(f"\n[ERROR] An unexpected error occurred during evaluation of {model_name} at {voltage}V: {e}")
            finally:
                if env:
                    env.close()
        
    # --- After all evaluations, save data and generate combined plots ---
    drive_save_path = f"/content/drive/MyDrive/DDC/Combined_Results/"
    os.makedirs(drive_save_path, exist_ok=True)

    json_plot_save_path = os.path.join(drive_save_path, "all_models_evaluation_data.json")
    print(f"\n--- Saving all plot data to {json_plot_save_path} ---")
    with open(json_plot_save_path, 'w') as f:
        json.dump(all_models_evaluation_data, f, indent=4)
    print("--- Plot data saved successfully! ---")

    json_metrics_save_path = os.path.join(drive_save_path, "all_models_evaluation_metrics.json")
    print(f"\n--- Saving all metrics data to {json_metrics_save_path} ---")
    with open(json_metrics_save_path, 'w') as f:
        json.dump(all_models_metrics_data, f, indent=4)
    print("--- Metrics data saved successfully! ---")

    if all_models_evaluation_data:
        plot_combined_summary(all_models_evaluation_data, drive_save_path)
