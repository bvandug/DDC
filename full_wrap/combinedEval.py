import numpy as np
from simulink_env import SimulinkEnv
from stable_baselines3 import TD3
import time

def evaluate_full_metrics(
    model, env, n_episodes=5,
    target_value=0.0, tolerance=0.1,
    stable_duration=0.1, sim_timestep=0.01
):
    all_rewards = []
    stabilisation_times = []
    steady_state_errors = []
    total_stable_times = []

    print("\nPer-Episode Metrics:")
    print("-" * 40)

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        t = 0.0
        ep_reward = 0.0
        stable_start_time = None
        last_state = None
        total_stable_time = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_reward += reward
            last_state = obs[0]

            # Check if within tolerance band
            if abs(last_state - target_value) < tolerance:
                total_stable_time += sim_timestep
                if stable_start_time is None:
                    stable_start_time = t
                elif t - stable_start_time >= stable_duration:
                    pass
            else:
                stable_start_time = None

            t += sim_timestep

        # Fallback if never stabilized
        stabilization_time = stable_start_time if stable_start_time is not None else t
        steady_error = abs(last_state - target_value)

        # Store metrics
        all_rewards.append(ep_reward)
        stabilisation_times.append(stabilization_time)
        steady_state_errors.append(steady_error)
        total_stable_times.append(total_stable_time)

        # Print individual episode metrics
        print(f"Episode {ep+1}:")
        print(f"  Total reward           : {ep_reward:.2f}")
        print(f"  Stabilisation time     : {stabilization_time:.2f} s")
        print(f"  Steady-state error     : {steady_error:.4f}")
        print(f"  Total stable time      : {total_stable_time:.2f} s")

    # Max reward calculation
    episode_length = env.max_episode_time
    max_total_reward = episode_length / sim_timestep
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    percent_max = 100 * mean_reward / max_total_reward

    # Summary
    print("\nSummary Metrics:")
    print("-" * 40)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f} (Max: {max_total_reward:.0f})")
    print(f"→ {percent_max:.1f}% of theoretical max reward")
    print(f"Mean Stabilisation Time : {np.mean(stabilisation_times):.2f} s")
    print(f"Mean Steady-State Error : {np.mean(steady_state_errors):.4f}")
    print(f"Mean Total Stable Time  : {np.mean(total_stable_times):.2f} s")

    return {
        "rewards": all_rewards,
        "stabilisation_times": stabilisation_times,
        "steady_state_errors": steady_state_errors,
        "total_stable_times": total_stable_times,
        "percent_of_max": percent_max,
    }


if __name__ == "__main__":
    env = SimulinkEnv()
    name = "td3_simulinker"
    print(f"Loading model {name}...")
    model = TD3.load(name)

    start = time.perf_counter()
    metrics = evaluate_full_metrics(
        model, env,
        n_episodes=5,
        target_value=0.0,
        tolerance=0.1,
        stable_duration=0.5,
        sim_timestep=0.01
    )
    elapsed = time.perf_counter() - start

    print(f"Total evaluation time: {elapsed:.2f} seconds")
