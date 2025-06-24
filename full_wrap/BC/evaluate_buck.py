import numpy as np
import time
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from BCSimTestEnv import BCSimulinkEnv # Import your environment
from stable_baselines3 import A2C       # Or the model class you are using

def run_evaluation(
    model,
    env,
    n_episodes=5,
    target_voltage=30.0,
    tolerance=0.5 #Set the tolerance to +-0.5V
):
    """
    Evaluates a trained model with a robust definition of stabilization.

    Args:
        model: The trained Stable Baselines3 model.
        env (VecNormalize): The vectorized and normalized environment.
        n_episodes (int): The number of episodes to run for evaluation.
        target_voltage (float): The target voltage for the episodes.
        tolerance (float): The voltage tolerance for stabilization.
    """

    # Initialize lists to store values for metric calculation
    all_rewards = []
    stabilisation_times = []
    stabilization_durations = []
    steady_state_errors = []
    overshoots = []
    undershoots = []

    env.env_method('set_goal_voltage', target_voltage) # Set the target voltage for the environment, default is 30V

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        
        episode_voltages, episode_times = [], [] # Reset the lists for each episode
        
        print(f"\n--- Running Evaluation Episode {ep + 1}/{n_episodes} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True) # Predict the action using the trained model
            obs, reward, terminated, info = env.step(action) # Step the environment with the action
            
            unnormalized_obs = env.get_original_obs() # Get the original observation from the environment (because the observations are normalized)
            current_voltage = unnormalized_obs[0][0] # Get the current voltage from the observation
            current_time = env.get_attr('current_time')[0] # Get the current time from the environment
            
            ep_reward += reward[0]
            episode_voltages.append(current_voltage)
            episode_times.append(current_time)
            
            if terminated[0] or info[0].get('TimeLimit.truncated', False):
                done = True # If the episode is terminated or truncated, set done to True

        # Metric Calculation
        # Exclude last data point because it is faulty (the agent finished running when the max_episode_time was reached but simulink returns one more observation)
        times_arr = np.array(episode_times[:-1]) #It is a solver artifact
        voltages_arr = np.array(episode_voltages[:-1])

        if len(times_arr) == 0: continue

        # --- CORRECTED STABILIZATION LOGIC ---
        has_stabilized = False
        stabilisation_time = times_arr[-1] # Default to full duration
        stable_time_for = 0.0
        steady_error = 0.0
        
        # Find all indices where the voltage is outside the tolerance band
        outside_tolerance_indices = np.where(np.abs(voltages_arr - target_voltage) > tolerance)[0]
        
        if len(outside_tolerance_indices) == 0:
            # If it was always inside the band, it stabilized at the start
            first_stable_index = 0
            has_stabilized = True
            stabilisation_time = times_arr[0]
        else:
            # Find the LAST time it was outside the band
            last_unstable_index = outside_tolerance_indices[-1]
            # The system is stable from the step *after* this last unstable point
            if last_unstable_index + 1 < len(times_arr):
                first_stable_index = last_unstable_index + 1
                stabilisation_time = times_arr[first_stable_index]
                has_stabilized = True
        
        if has_stabilized:
            steady_state_voltages = voltages_arr[first_stable_index:]
            steady_error = np.mean(steady_state_voltages - target_voltage)
            stable_time_for = times_arr[-1] - stabilisation_time

        overshoot = max(0, np.max(voltages_arr) - target_voltage)
        undershoot = max(0, target_voltage - np.min(voltages_arr))

        all_rewards.append(ep_reward)
        stabilisation_times.append(stabilisation_time)
        stabilization_durations.append(stable_time_for)
        steady_state_errors.append(steady_error if has_stabilized else np.nan) # Use NaN if not stable
        overshoots.append(overshoot)
        undershoots.append(undershoot)

        print(f"  Episode Reward         : {ep_reward:.2f}")
        print(f"  Stabilisation Time     : {stabilisation_time * 1000:.2f} ms" if has_stabilized else "  Stabilisation Time     : Not achieved")
        print(f"  Stabilisation Duration : {stable_time_for * 1000:.2f} ms")
        print(f"  Steady-State Error     : {steady_error:+.4f} V" if has_stabilized else "  Steady-State Error     : N/A")
        print(f"  Overshoot              : {overshoot:.4f} V")
        print(f"  Undershoot             : {undershoot:.4f} V")

    # --- Summary Statistics ---
    print("\n--- Evaluation Summary ---")
    print("-" * 40)
    print(f"Mean Reward             : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean Stabilisation Time : {np.mean(stabilisation_times) * 1000:.2f} ms")
    print(f"Mean Stabilisation Dur. : {np.mean(stabilization_durations) * 1000:.2f} ms")
    print(f"Mean Steady-State Error : {np.nanmean(steady_state_errors):.4f} V") # nanmean ignores runs that didn't stabilize
    print(f"Mean Overshoot          : {np.mean(overshoots):.4f} V")
    print(f"Mean Undershoot         : {np.mean(undershoots):.4f} V")
    print("-" * 40)

    return {
        "rewards": all_rewards,
        "stabilisation_times": stabilisation_times,
        "stabilization_durations": stabilization_durations,
        "steady_state_errors": steady_state_errors,
        "overshoots": overshoots,
        "undershoots": undershoots,
    }

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_SAVE_PATH = "a2c_bc_model_final.zip"
    STATS_PATH = "vec_normalize_stats_final.pkl"
    TARGET_VOLTAGE = 30.0
    N_EVAL_EPISODES = 5

    # --- Setup ---
    print("--- Setting up for STATISTICAL evaluation ---")
    env = None
    try:
        env_fn = lambda: BCSimulinkEnv(
            model_name="bcSim",
            enable_plotting=True,
            target_voltage=TARGET_VOLTAGE
        )
        env = DummyVecEnv([env_fn])
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False
        
        model = A2C.load(MODEL_SAVE_PATH, env=env)
        
        # --- Run Evaluation ---
        start_time = time.perf_counter()
        metrics = run_evaluation(
            model=model,
            env=env,
            n_episodes=N_EVAL_EPISODES,
            target_voltage=TARGET_VOLTAGE,
            tolerance=0.5
        )
        end_time = time.perf_counter()
        print(f"\nTotal evaluation time: {end_time - start_time:.2f} seconds")

    except FileNotFoundError as e:
        print(f"\n[ERROR] A required file was not found: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if env:
            env.close()
