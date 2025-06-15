import numpy as np
from simulink_env import SimulinkEnv
from stable_baselines3 import TD3,A2C
import time
import matplotlib.pyplot as plt


def evaluate_full_metrics(
    model, env, n_episodes=5,
    target_value=0.0, tolerance=0.1,
    stable_duration=0.1, sim_timestep=0.01
):
    all_rewards = []
    stabilisation_times = []
    steady_state_errors = []
    total_stable_times = []
    all_thetas = []
    all_theta_vs = []
    all_us = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        t = 0.0
        ep_reward = 0.0
        stable_start_time = None
        last_state = None
        total_stable_time = 0.0

        episode_thetas = []
        episode_theta_vs = []
        episode_us = []
        t_vals = []

        # === Live Plot Setup ===
        plt.ion()
        fig, axs = plt.subplots(4, 1, figsize=(8, 10))
        theta_line, = axs[0].plot([], [], label="theta (deg)")
        theta_v_line, = axs[1].plot([], [], label="theta_v (deg/s)")
        u_line, = axs[2].plot([], [], label="u (N)")

        for ax, title, ylabel in zip(
            axs[:3],
            ["Theta (deg)", "Theta Velocity (deg/s)", "Control Input (N)"],
            ["°", "°/s", "N"]
        ):
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()

        # === Pendulum Drawing ===
        ax_pendulum = axs[3]
        ax_pendulum.set_xlim(-1.5, 1.5)
        ax_pendulum.set_ylim(-1.5, 1.5)
        ax_pendulum.set_aspect("equal")
        pendulum_line, = ax_pendulum.plot([], [], lw=4)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_reward += reward
            theta, theta_v = obs
            u = action[0]
            last_state = theta

            theta_deg = np.degrees(theta)
            theta_v_deg = np.degrees(theta_v)

            episode_thetas.append(theta_deg)
            episode_theta_vs.append(theta_v_deg)
            episode_us.append(u)
            t_vals.append(t)

            # === Update Plots ===
            theta_line.set_data(t_vals, episode_thetas)
            theta_v_line.set_data(t_vals, episode_theta_vs)
            u_line.set_data(t_vals, episode_us)

            for ax in axs[:3]:
                ax.relim()
                ax.autoscale_view()

            # === Update Pendulum Animation ===
            cart_x = 0.0
            L = 1.0
            pend_x = cart_x + L * np.sin(theta)
            pend_y = -L * np.cos(theta)
            pendulum_line.set_data([cart_x, pend_x], [0.0, pend_y])
            ax_pendulum.set_title(f"Pendulum Angle: {theta_deg:.1f}°")

            plt.pause(0.001)

            # === Stability check ===
            if abs(theta - target_value) < tolerance:
                total_stable_time += sim_timestep
                if stable_start_time is None:
                    stable_start_time = t
            else:
                stable_start_time = None

            t += sim_timestep

        stabilization_time = stable_start_time if stable_start_time is not None else t
        steady_error = abs(np.degrees(last_state - target_value))  # now in degrees

        all_rewards.append(ep_reward)
        stabilisation_times.append(stabilization_time)
        steady_state_errors.append(steady_error)
        total_stable_times.append(total_stable_time)
        all_thetas.append(episode_thetas)
        all_theta_vs.append(episode_theta_vs)
        all_us.append(episode_us)

        print(f"\nEpisode {ep+1}:")
        print(f"  Total reward           : {ep_reward:.2f}")
        print(f"  Stabilisation time     : {stabilization_time:.2f} s")
        print(f"  Steady-state error     : {steady_error:.2f}°")
        print(f"  Total stable time      : {total_stable_time:.2f} s")

        # === Save and close figure ===
        plt.ioff()
        filename = f"episode_{ep+1:02d}_plot1.png"
        # plt.savefig(filename)
        # print(f"Saved plot to {filename}")
        plt.close(fig)

    # === Summary ===
    episode_length = env.max_episode_time
    max_total_reward = episode_length / sim_timestep
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    percent_max = 100 * mean_reward / max_total_reward

    print("\nSummary Metrics:")
    print("-" * 40)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f} (Max: {max_total_reward:.0f})")
    print(f"→ {percent_max:.1f}% of theoretical max reward")
    print(f"Mean Stabilisation Time : {np.mean(stabilisation_times):.2f} s")
    print(f"Mean Steady-State Error : {np.mean(steady_state_errors):.2f}°")
    print(f"Mean Total Stable Time  : {np.mean(total_stable_times):.2f} s")

    return {
        "rewards": all_rewards,
        "stabilisation_times": stabilisation_times,
        "steady_state_errors": steady_state_errors,
        "total_stable_times": total_stable_times,
        "percent_of_max": percent_max,
        "theta": all_thetas,
        "theta_v": all_theta_vs,
        "u": all_us,
    }


if __name__ == "__main__":
    env = SimulinkEnv()
    name = "best_model_a2c"
    print(f"Loading model {name}...")
    model = A2C.load(name)

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
