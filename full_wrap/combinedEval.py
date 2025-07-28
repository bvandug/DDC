import os
import numpy as np
from ip_simulink_env import SimulinkEnv
from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO, DQN
import matplotlib.pyplot as plt

def evaluate_full_metrics(
    model, env, n_episodes=5,
    target_value=0.0, tolerance=0.1,
    stable_duration=0.1, sim_timestep=0.01,
    live_plot=False,
    save_plots=True,
    output_dir='plots'
):
    """
    Evaluate a control model over multiple episodes and generate metrics, plots, and logs.

    Args:
        model: Trained control policy.
        env: Simulation environment.
        n_episodes: Number of episodes to run.
        target_value: Desired upright angle (radians).
        tolerance: Threshold for stability (radians).
        stable_duration: Not used but reserved for future.
        sim_timestep: Simulation time step (s).
        live_plot: If True, shows real-time plots.
        save_plots: If True, saves episode and summary plots and logs.
        output_dir: Directory to save outputs.

    Returns:
        Metrics dict: rewards, stabilisation_times, steady_state_errors,
        overshoots, total_stable_times, percent_of_max.
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    log_lines = []

    all_rewards = []
    stabilisation_times = []
    steady_state_errors = []
    total_stable_times = []
    all_thetas = []
    all_theta_vs = []
    all_us = []
    overshoots = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        t = 0.0
        ep_reward = 0.0
        stable_start = None
        total_stable = 0.0
        episode_thetas, episode_theta_vs, episode_us, t_vals = [], [], [], []

        # live plotting setup
        if live_plot:
            plt.ion()
            fig_live, axs_live = plt.subplots(4, 1, figsize=(8, 10))
            theta_line, = axs_live[0].plot([], [], label='theta (deg)')
            theta_v_line, = axs_live[1].plot([], [], label='theta_v (deg/s)')
            u_line, = axs_live[2].plot([], [], label='u (N)')
            for ax, title, ylabel in zip(
                axs_live[:3],
                ['Theta (deg)', 'Theta Velocity (deg/s)', 'Control Input'],
                ['°', '°/s', 'N']
            ):
                ax.set(title=title, xlabel='Time (s)', ylabel=ylabel)
                ax.grid(); ax.legend()
            axp = axs_live[3]
            axp.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect='equal')
            pend_line, = axp.plot([], [], lw=4)

        # episode loop
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            action = np.atleast_1d(action)
            ep_reward += reward
            theta, theta_v = obs
            theta_deg = np.degrees(theta)
            theta_v_deg = np.degrees(theta_v)
            u = action[0]

            episode_thetas.append(theta_deg)
            episode_theta_vs.append(theta_v_deg)
            episode_us.append(u)
            t_vals.append(t)

            if live_plot:
                theta_line.set_data(t_vals, episode_thetas)
                theta_v_line.set_data(t_vals, episode_theta_vs)
                u_line.set_data(t_vals, episode_us)
                for ax in axs_live[:3]:
                    ax.relim(); ax.autoscale_view()
                pend_line.set_data([0, np.sin(theta)], [0, -np.cos(theta)])
                axs_live[3].set_title(f'Angle: {theta_deg:.1f}°')
                plt.pause(0.001)

            if abs(theta - target_value) < tolerance:
                total_stable += sim_timestep
                if stable_start is None:
                    stable_start = t
            else:
                stable_start = None

            t += sim_timestep

        if live_plot:
            plt.ioff(); plt.close(fig_live)

        stab_time = stable_start if stable_start is not None else t
        steady_err = abs(np.degrees(theta - target_value))
        over = max(abs(v) for v in episode_thetas)

        all_rewards.append(ep_reward)
        stabilisation_times.append(stab_time)
        steady_state_errors.append(steady_err)
        total_stable_times.append(total_stable)
        all_thetas.append((t_vals, episode_thetas))
        all_theta_vs.append((t_vals, episode_theta_vs))
        all_us.append((t_vals, episode_us))
        overshoots.append(over)

        # logging
        lines = [f'Episode {ep+1}:',
                 f'  Total reward           : {ep_reward:.2f}',
                 f'  Stabilisation time     : {stab_time:.2f} s',
                 f'  Steady-state error     : {steady_err:.2f}°',
                 f'  Overshoot magnitude    : {over:.2f}°',
                 f'  Total stable time      : {total_stable:.2f} s']
        for l in lines:
            print(l)
            log_lines.append(l)

        # save per-episode plot
        if save_plots:
            fname = os.path.join(output_dir, f'{model.__class__.__name__}_episode_{ep+1}.png')
            fig_ep, ax_ep = plt.subplots(figsize=(6, 3))
            ax_ep.plot(t_vals, episode_thetas, label=f'Episode {ep+1}')
            # add tolerance band in degrees
            tol_deg = np.degrees(tolerance)
            ax_ep.axhline( tol_deg, linestyle='--', color='gray', label=f'+{tol_deg:.1f}° tol' )
            ax_ep.axhline(-tol_deg, linestyle='--', color='gray', label=f'-{tol_deg:.1f}° tol' )
            ax_ep.axhline(0,  linestyle='--', color='red')
            ax_ep.set(title=f'Theta Trajectory - Episode {ep+1}',
                    xlabel='Time (s)', ylabel='Theta (deg)')
            ax_ep.grid()
            ax_ep.legend(loc='lower right')
            fig_ep.tight_layout()
            fig_ep.savefig(fname, bbox_inches='tight')
            plt.close(fig_ep)

    # summary
    max_r = env.max_episode_time / sim_timestep
    mean_r, std_r = np.mean(all_rewards), np.std(all_rewards)
    percent_max = 100 * mean_r / max_r
    summary_lines = ['', 'Summary Metrics:', '-'*40,
                     f'Mean Reward: {mean_r:.2f} ± {std_r:.2f} (Max: {max_r:.0f})',
                     f'→ {percent_max:.1f}% of theoretical max reward',
                     f'Mean Stabilisation Time : {np.mean(stabilisation_times):.2f} s',
                     f'Mean Steady-State Error : {np.mean(steady_state_errors):.2f}°',
                     f'Mean Overshoot Magnitude: {np.mean(overshoots):.2f}°',
                     f'Mean Total Stable Time  : {np.mean(total_stable_times):.2f} s']
    for l in summary_lines:
        print(l)
        log_lines.append(l)

       # save final plots and logs
    if save_plots:
        # combined subplot (legends on right; full degree labels)
        fig_all, axs_all = plt.subplots(3, 1, figsize=(8, 10))
        labels = [f'Episode {i+1}' for i in range(len(all_thetas))]
        for idx, ((t_v, th), (_, tv), (_, u)) in enumerate(zip(all_thetas, all_theta_vs, all_us)):
            axs_all[0].plot(t_v, th, label=labels[idx])
            axs_all[1].plot(t_v, tv, label=labels[idx])
            axs_all[2].plot(t_v, u, label=labels[idx])
        titles = ['Theta (deg)', 'Theta Velocity (deg/s)', 'Control Input (N)']
        ylabels = ['Theta (deg)', 'Theta Velocity (deg/s)', 'Control Input (N)']
        for i, ax in enumerate(axs_all):
            ax.set(title=titles[i], xlabel='Time (s)', ylabel=ylabels[i])
            ax.grid(True)
            loc = 'upper right' if i < 2 else 'lower right'
            ax.legend(loc=loc)
        fig_all.tight_layout()
        fig_all.savefig(os.path.join(output_dir, 'combined_subplot.png'), bbox_inches='tight')
        plt.close(fig_all)

            # theta-only final (degrees labeled, legend on right, with tolerance bands)
        tol_deg = np.degrees(tolerance)
        fig_th, ax_th = plt.subplots(figsize=(8, 4))
        for idx, (t_v, th) in enumerate(all_thetas):
            ax_th.plot(t_v, th, label=labels[idx])
        # zero line
        ax_th.axhline(0, linestyle='--', color='red', label='0°')
        # tolerance bands
        ax_th.axhline( tol_deg, linestyle='--', color='gray', label=f'+{tol_deg:.1f}° tol')
        ax_th.axhline(-tol_deg, linestyle='--', color='gray', label=f'-{tol_deg:.1f}° tol')
        ax_th.set(
            title='All Episodes: Theta (deg)',
            xlabel='Time (s)',
            ylabel='Theta (deg)'
        )
        ax_th.grid(True)
        ax_th.legend(loc='upper right')
        fig_th.tight_layout()
        fig_th.savefig(os.path.join(output_dir, 'theta_plot.png'), bbox_inches='tight')
        plt.close(fig_th)

        # write results log
        log_file = os.path.join(output_dir, f'{model.__class__.__name__}_results.txt')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_lines))

    return {
        'rewards': all_rewards,
        'stabilisation_times': stabilisation_times,
        'steady_state_errors': steady_state_errors,
        'overshoots': overshoots,
        'total_stable_times': total_stable_times,
        'percent_of_max': percent_max
    }

if __name__ == '__main__':
    env = SimulinkEnv()
    models = {
        'A2C': 'ip_jax/jax/a2c/best_model.zip',
        'SAC': 'ip_jax/jax/sac/best_model.zip',
        'DQN': 'ip_jax/jax/dqn/best_model.zip',
        'PPO': 'ip_jax/jax/ppo/best_model.zip',
        'DDPG': 'ip_jax/jax/ddpg/best_model.zip',
        'TD3': 'ip_jax/jax/td3/best_model.zip'
    }
    results = {}
    for name, path in models.items():
        print(f'\n=== Evaluating {name} ===')
        ModelClass = globals()[name]
        if name in ('PPO'):
            model = ModelClass.load(path, device='cpu')
        else:
            model = ModelClass.load(path)
        results[name] = evaluate_full_metrics(
            model, env,
            live_plot=False,
            save_plots=True,
            output_dir=os.path.join('plots', name)
        )

    print('\nAll evaluations complete.')
