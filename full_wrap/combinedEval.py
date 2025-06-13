import numpy as np
from simulink_env import SimulinkEnv
from stable_baselines3 import TD3
import time

def evaluate_full_metrics(
    model, env, n_episodes=5,
    target_value=0.0, tolerance=0.01,
    stable_duration=0.5, sim_timestep=0.01
):
    all_rewards = []
    stabilisation_times = []
    steady_state_errors = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        t = 0.0
        ep_reward = 0.0
        stable_start_time = None
        last_state = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_reward += reward
            last_state = obs[0]  # Customize if needed

            # Stabilization detection
            if abs(last_state - target_value) < tolerance:
                if stable_start_time is None:
                    stable_start_time = t
                elif t - stable_start_time >= stable_duration:
                    # Stabilized — break early if desired
                    pass
            else:
                stable_start_time = None

            t += sim_timestep

        # Store metrics
        all_rewards.append(ep_reward)
        stabilisation_times.append(stable_start_time if stable_start_time is not None else t)
        steady_state_errors.append(abs(last_state - target_value))

    # Compute and print summary
    print(f"Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Mean Stabilisation Time: {np.mean(stabilisation_times):.2f} s")
    print(f"Mean Steady-State Error: {np.mean(steady_state_errors):.4f}")

    return {
        "rewards": all_rewards,
        "stabilisation_times": stabilisation_times,
        "steady_state_errors": steady_state_errors
    }

if __name__ == "__main__":
    env   = SimulinkEnv()
    model = TD3.load("td3_simulinker")

    start = time.perf_counter()
    metrics = evaluate_full_metrics(
        model, env,
        n_episodes=5,
        target_value=0.0,
        tolerance=0.01,
        stable_duration=0.5,
        sim_timestep=0.01
    )
    elapsed = time.perf_counter() - start

    print(f"Total evaluation time: {elapsed:.2f} seconds")
