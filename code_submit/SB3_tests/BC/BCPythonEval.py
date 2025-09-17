import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC, A2C, TD3, PPO, DDPG, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from BCPythonEnv import BuckConverterEnv, DiscretizeActionWrapper


def plot_and_save_summary(all_episode_data, target_voltage, tolerance,
                          model_type, plot_save_base):
    """Generates and saves summary plots from evaluation episodes.

    This function creates two plots for visualizing performance:
    1. A full view of the entire voltage trajectory.
    2. A zoomed-in view on the target voltage for steady-state analysis.

    Args:
        all_episode_data (list): A list of tuples, where each tuple contains
            (times_arr, voltages_arr) for a single evaluation episode.
        target_voltage (float): The target voltage for the evaluation.
        tolerance (float): The voltage tolerance band (±V) used for plotting.
        model_type (str): A string identifying the model (e.g., 'SAC').
        plot_save_base (str): The base path and filename for saved plots.
    """
    # Generate Full View Plot
    full_view_path = f"{plot_save_base}_full_view.png"
    print(f"\n--- Saving full view plot to {full_view_path} ---")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(15, 8))

    for i, (times, voltages) in enumerate(all_episode_data):
        ax1.plot(times * 1000, voltages, label=f'Episode {i + 1} Voltage',
                 alpha=0.8)

    ax1.axhline(y=target_voltage, color='r', linestyle='--', linewidth=2,
                label='Target Voltage')
    ax1.axhspan(target_voltage - tolerance, target_voltage + tolerance,
                color='red', alpha=0.1, label=f'Tolerance (±{tolerance}V)')

    ax1.set_title(
        f'{model_type} Agent Performance (Target: {target_voltage}V)',
        fontsize=16, weight='bold'
    )
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Output Voltage (V)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    if all_episode_data and any(v.size > 0 for _, v in all_episode_data):
        voltages = [v for _, v in all_episode_data if v.size > 0]
        min_volt = np.min([np.min(v) for v in voltages])
        max_volt = np.max([np.max(v) for v in voltages])
        ax1.set_ylim(
            bottom=min(target_voltage - 5, min_volt - 1),
            top=max(target_voltage + 5, max_volt + 1)
        )

    fig1.savefig(full_view_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("--- Full view plot saved successfully! ---")

    # Generate Zoomed-in Plot
    zoomed_view_path = f"{plot_save_base}_zoomed_view.png"
    print(f"\n--- Saving zoomed view plot to {zoomed_view_path} ---")

    fig2, ax2 = plt.subplots(figsize=(15, 8))

    for i, (times, voltages) in enumerate(all_episode_data):
        ax2.plot(times * 1000, voltages, label=f'Episode {i + 1} Voltage',
                 alpha=0.8, marker='.', markersize=2, linestyle='-')

    ax2.axhline(y=target_voltage, color='r', linestyle='--', linewidth=2,
                label='Target Voltage')
    ax2.axhspan(target_voltage - tolerance, target_voltage + tolerance,
                color='red', alpha=0.1, label=f'Tolerance (±{tolerance}V)')
    ax2.set_title(
        f'{model_type} Agent Performance - Zoomed View '
        f'(Target: {target_voltage}V)',
        fontsize=16, weight='bold'
    )
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Output Voltage (V)', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_ylim(bottom=target_voltage - 2.5, top=target_voltage + 2.5)

    fig2.savefig(zoomed_view_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("--- Zoomed view plot saved successfully! ---")


def run_evaluation(model, env, n_episodes=5, target_voltage=30.0,
                   tolerance=0.5):
    """Runs a full evaluation of a model and calculates performance metrics.

    Args:
        model: The trained Stable Baselines3 model to evaluate.
        env: The vectorized and normalized evaluation environment.
        n_episodes (int): The number of episodes to run the evaluation for.
        target_voltage (float): The target voltage for the evaluation episodes.
        tolerance (float): The voltage tolerance band (±V) for calculating KPIs.

    Returns:
        list: A list of tuples, where each tuple contains (times_arr,
        voltages_arr) for a single episode, suitable for plotting.
    """
    all_rewards, stabilisation_times, stabilization_durations = [], [], []
    steady_state_errors, overshoots, undershoots = [], [], []
    all_episode_plot_data = []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        episode_voltages, episode_times = [], []

        print(f"\n--- Running Evaluation Episode {ep + 1}/{n_episodes} ---")

        while True:
            action, _ = model.predict(obs, deterministic=True)

            # Capture the state *before* taking the step for accurate logging.
            unnormalized_obs = env.get_original_obs()
            current_time = env.get_attr('total_sim_time')[0]

            obs, reward, dones, infos = env.step(action)

            ep_reward += reward[0]
            # Use the true voltage from the info dict for plotting
            episode_voltages.append(infos[0]['true_voltage'])
            episode_times.append(current_time)

            if dones[0]:
                break

        times_arr = np.array(episode_times)
        voltages_arr = np.array(episode_voltages)

        if len(times_arr) == 0:
            continue

        all_episode_plot_data.append((times_arr, voltages_arr))

        # --- Performance Metric Calculation ---
        has_stabilized = False
        stabilisation_time = times_arr[-1]

        outside_indices = np.where(
            np.abs(voltages_arr - target_voltage) > tolerance
        )[0]

        if len(outside_indices) == 0:
            first_stable_index = 0
            has_stabilized = True
            stabilisation_time = times_arr[0] if len(times_arr) > 0 else 0
        else:
            last_unstable_index = outside_indices[-1]
            if last_unstable_index + 1 < len(times_arr):
                first_stable_index = last_unstable_index + 1
                stabilisation_time = times_arr[first_stable_index]
                has_stabilized = True

        if has_stabilized:
            steady_voltages = voltages_arr[first_stable_index:]
            steady_error = np.mean(steady_voltages - target_voltage)
            stable_time_for = times_arr[-1] - stabilisation_time
        else:
            stable_time_for = 0.0
            steady_error = np.nan

        overshoot = max(0, np.max(voltages_arr) - target_voltage)

        first_cross_indices = np.where(voltages_arr >= target_voltage)[0]
        if len(first_cross_indices) > 0:
            voltages_after = voltages_arr[first_cross_indices[0]:]
            undershoot = max(0, target_voltage - np.min(voltages_after))
        else:
            undershoot = 0.0

        all_rewards.append(ep_reward)
        stabilisation_times.append(stabilisation_time)
        stabilization_durations.append(stable_time_for)
        steady_state_errors.append(steady_error)
        overshoots.append(overshoot)
        undershoots.append(undershoot)

        # Print per-episode results.
        print(f"  Episode Reward         : {ep_reward:.2f}")
        if has_stabilized:
            print(f"  Stabilisation Time     : "
                  f"{stabilisation_time * 1000:.2f} ms")
            print(f"  Steady-State Error     : {steady_error:+.4f} V")
        else:
            print("  Stabilisation Time     : Not achieved")
            print("  Steady-State Error     : N/A")
        print(f"  Stabilisation Duration : {stable_time_for * 1000:.2f} ms")
        print(f"  Overshoot              : {overshoot:.4f} V")
        print(f"  Undershoot             : {undershoot:.4f} V")

    # Print Overall Summary Metrics
    print("\n--- Evaluation Summary ---")
    print("-" * 40)
    print(f"Mean Reward              : {np.mean(all_rewards):.2f} "
          f"± {np.std(all_rewards):.2f}")
    print(f"Mean Stabilisation Time  : "
          f"{np.mean(stabilisation_times) * 1000:.2f} ms")
    print(f"Mean Stabilisation Dur.  : "
          f"{np.mean(stabilization_durations) * 1000:.2f} ms")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        print(f"Mean Steady-State Error  : "
              f"{np.nanmean(steady_state_errors):.4f} V")
    print(f"Mean Overshoot           : {np.mean(overshoots):.4f} V")
    print(f"Mean Undershoot          : {np.mean(undershoots):.4f} V")
    print("-" * 40)

    return all_episode_plot_data


if __name__ == '__main__':
    MODELS_TO_EVALUATE = ['A2C', 'PPO', 'SAC', 'TD3', 'DDPG', 'DQN']
    NOISE_LEVELS = [0, 0.001, 0.01, 0.1]
    EVALUATION_VOLTAGES = [30.0]
    N_EVAL_EPISODES = 3
    MAX_EPISODE_STEPS = 2001

    # Define local paths for loading models and saving results
    BASE_MODEL_PATH = "models"
    RESULTS_SAVE_DIR = "PY_BC_Results"

    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

    # Main Evaluation Loop
    for model_type in MODELS_TO_EVALUATE:
        for noise in NOISE_LEVELS:
            model_folder = os.path.join(
                BASE_MODEL_PATH,
                model_type,
                f"{model_type}_Seed_42_Noise_{noise}"
            )
            MODEL_LOAD_PATH = os.path.join(model_folder, "final_model.zip")
            STATS_LOAD_PATH = os.path.join(model_folder,
                                           "vec_normalize.pkl")

            if not os.path.exists(MODEL_LOAD_PATH) or \
               not os.path.exists(STATS_LOAD_PATH):
                print(f"\n[SKIP] Files not found for {model_type} with noise "
                      f"{noise}. Searched in: {model_folder}")
                continue

            for voltage in EVALUATION_VOLTAGES:
                print("\n" + "=" * 80)
                print(f"--- EVALUATING {model_type} | Noise: {noise} | "
                      f"Voltage: {voltage}V ---")
                print("=" * 80 + "\n")

                plot_save_folder = os.path.join(RESULTS_SAVE_DIR,
                                                model_type)
                os.makedirs(plot_save_folder, exist_ok=True)
                plot_save_base_name = f"eval_noise_{noise}_voltage_{voltage}"
                PLOT_SAVE_BASE = os.path.join(plot_save_folder,
                                              plot_save_base_name)

                print(f"Loading model from: {model_folder}")
                print(f"Saving plots to:    {plot_save_folder}")

                env = None
                try:
                    def create_env_fn(current_noise):
                        def _init():
                            return BuckConverterEnv(
                                render_mode=None,
                                use_randomized_goal=False,
                                fixed_goal_voltage=voltage,
                                max_episode_steps=MAX_EPISODE_STEPS,
                                voltage_noise_std=current_noise
                            )
                        return _init

                    base_env_fn = create_env_fn(noise)

                    # Wrap the environment for DQN's discrete action space.
                    if model_type == 'DQN':
                        env_fn = lambda: DiscretizeActionWrapper(
                            base_env_fn(), n_bins=17
                        )
                    else:
                        env_fn = base_env_fn

                    env = DummyVecEnv([env_fn])
                    env = VecNormalize.load(STATS_LOAD_PATH, env)
                    env.training = False
                    env.norm_reward = False

                    # Dynamically load the correct model class.
                    model_class_map = {
                        'SAC': SAC, 'TD3': TD3, 'A2C': A2C,
                        'DDPG': DDPG, 'PPO': PPO, 'DQN': DQN
                    }
                    model_class = model_class_map.get(model_type)

                    if model_class is None:
                        raise ValueError(f"Model type '{model_type}' "
                                         "not recognized.")

                    model = model_class.load(MODEL_LOAD_PATH, env=env)

                    start_time = time.perf_counter()
                    plot_data = run_evaluation(
                        model=model,
                        env=env,
                        n_episodes=N_EVAL_EPISODES,
                        target_voltage=voltage,
                        tolerance=0.5
                    )
                    end_time = time.perf_counter()
                    print(f"\nTotal evaluation time for {voltage}V: "
                          f"{end_time - start_time:.2f} seconds")

                    if plot_data:
                        plot_and_save_summary(
                            all_episode_data=plot_data,
                            target_voltage=voltage,
                            tolerance=0.5,
                            model_type=f"{model_type}_noise_{noise}",
                            plot_save_base=PLOT_SAVE_BASE
                        )

                except FileNotFoundError as e:
                    print(f"\n[ERROR] A required file was not found: {e}")
                except Exception as e:
                    print(f"\n[UNEXPECTED ERROR] {e}")
                finally:
                    if env:
                        env.close()
                        print(f"\n--- Evaluation for {voltage}V complete. ---")
