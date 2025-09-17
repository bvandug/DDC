import os
import argparse
import numpy as np
from cp_simulink_env import SimulinkEnv, DiscretizedActionWrapper
from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO, DQN
import matplotlib.pyplot as plt
import control as ctrl

def signed_over_from_stepinfo(theta_deg: np.ndarray,
                              t: np.ndarray,
                              setpoint_deg: float,
                              debug: bool = False,
                              exclude_initial_s: float = 0.10) -> tuple[dict, dict]:
    """ Compute signed over/undershoot metrics from a step response (deg).

            Given angle samples in **degrees** and aligned time stamps (seconds),
            this function derives a set of over/undershoot metrics relative to a
            target setpoint (also in degrees). The sign convention encodes the
            **type** of transient:
            - **Positive** `signed_over_setpoint` ⇒ the response *crossed* the
            setpoint (classical overshoot).
            - **Negative** `signed_over_setpoint` ⇒ the response did *not* cross
            the setpoint (undershoot).

            Parameters
            ----------
            theta_deg : np.ndarray
                Measured angle trajectory in **degrees**.
            t : np.ndarray
                Time stamps (seconds), same length as `theta_deg`.
            setpoint_deg : float
                Desired setpoint in **degrees** (commonly 0.0).
            debug : bool, optional
                If True, prints intermediate calculations for inspection.
            exclude_initial_s : float, optional
                Reserved for future use (currently unused).

            Returns
            -------
            tuple[dict, dict]
                `(S, metrics)` where:
                - `S` : dict returned by `control.step_info(y, T=t)` if available,
                else `{}`.
                - `metrics` : dict with:
                    * `pos_over_setpoint` : float
                        Max positive error relative to setpoint.
                    * `neg_under_setpoint` : float
                        Min negative error relative to setpoint.
                    * `signed_over_setpoint` : float
                        Signed magnitude of the *first* significant peak after
                        initial crossing decision; positive if crossed setpoint,
                        negative otherwise.
                    * `pos_over_final` : float
                        Overshoot above estimated final value (non-negative).
                    * `neg_under_final` : float
                        Undershoot below estimated final value (non-positive).
                    * `signed_over_final` : float
                        Larger-magnitude of the two above, with its sign.
                    * `y_ss` : float
                        Estimated steady-state value (deg), computed as the mean of
                        the last ~10% (min 3 samples).

            Notes
            -----
            - Crossing detection is based on sign changes of `e(t) = y(t) - setpoint`.
            - Peaks are located via slope sign changes on the relevant segment.
            - On invalid inputs (length mismatch or too few samples), zero-like
            metrics are returned and `y_ss` falls back to the last sample.
        """

    t = np.asarray(t, dtype=float)
    y = np.asarray(theta_deg, dtype=float)
    if len(t) != len(y) or len(y) < 3:
        return {}, {
            'pos_over_setpoint': 0.0, 'neg_under_setpoint': 0.0, 'signed_over_setpoint': 0.0,
            'pos_over_final': 0.0, 'neg_under_final': 0.0, 'signed_over_final': 0.0,
            'y_ss': float(y[-1]) if len(y) else 0.0
        }

    e = y - setpoint_deg
    peak_value = 0.0  # Intermediate variable to hold the raw peak value

    sign_changes = np.where(np.diff(np.sign(e)))[0]

    if sign_changes.size > 0:
        # --- SCENARIO 1: OVERSHOOT LOGIC (Setpoint was crossed) ---
        first_cross_idx = sign_changes[0]
        e_after_cross = e[first_cross_idx + 1:]

        if e_after_cross.size > 2:
            de = np.diff(e_after_cross)
            s = np.sign(de)
            s[s == 0] = 1
            maxima_idx_rel = np.where(np.diff(s) < 0)[0] + 1
            minima_idx_rel = np.where(np.diff(s) > 0)[0] + 1
            first_max_idx_rel = maxima_idx_rel[0] if maxima_idx_rel.size > 0 else None
            first_min_idx_rel = minima_idx_rel[0] if minima_idx_rel.size > 0 else None

            if first_max_idx_rel is not None and first_min_idx_rel is not None:
                peak_value = float(e_after_cross[first_max_idx_rel]) if first_max_idx_rel < first_min_idx_rel else float(e_after_cross[first_min_idx_rel])
            elif first_max_idx_rel is not None:
                peak_value = float(e_after_cross[first_max_idx_rel])
            elif first_min_idx_rel is not None:
                peak_value = float(e_after_cross[first_min_idx_rel])
    else:
        # --- SCENARIO 2: UNDERSHOOT LOGIC (Setpoint was NOT reached) ---
        if e.size > 2:
            de = np.diff(e)
            s = np.sign(de)
            s[s == 0] = 1
            maxima_idx = np.where(np.diff(s) < 0)[0] + 1
            minima_idx = np.where(np.diff(s) > 0)[0] + 1

            if maxima_idx.size > 0:
                peak_value = float(e[maxima_idx[0]])
            elif minima_idx.size > 0:
                peak_value = float(e[minima_idx[0]])

    if sign_changes.size > 0:
        # It was an OVERSHOOT, so the result must be POSITIVE.
        signed_over_setpoint = abs(peak_value)
    else:
        # It was an UNDERSHOOT, so the result must be NEGATIVE.
        signed_over_setpoint = -abs(peak_value)

    # The rest of the metrics are calculated on the full signal for context.
    de_full = np.diff(e)
    s_full = np.sign(de_full)
    s_full[s_full == 0] = 1
    maxima_idx_full = np.where(np.diff(s_full) < 0)[0] + 1
    minima_idx_full = np.where(np.diff(s_full) > 0)[0] + 1
    pos_over_setpoint = float(np.max(e[maxima_idx_full])) if maxima_idx_full.size else 0.0
    neg_under_setpoint = float(np.min(e[minima_idx_full])) if minima_idx_full.size else 0.0

    tail = max(3, int(0.1 * len(y)))
    y_ss = float(np.mean(y[-tail:]))
    e_final = y_ss - setpoint_deg
    pos_over_final = float(max(0.0, pos_over_setpoint - e_final))
    neg_under_final = float(min(0.0, neg_under_setpoint - e_final))
    signed_over_final = pos_over_final if pos_over_final >= abs(neg_under_final) else neg_under_final
    S = {}
    try:
        S = ctrl.step_info(y, T=t)
    except Exception:
        pass

    if debug:
        print("Debug overshoot calc (V4 Logic):")
        print(f"  Setpoint={setpoint_deg:.2f}°, Estimated y_ss={y_ss:.2f}°")
        print(f"  Overall Peaks: Max positive={pos_over_setpoint:.2f}°, Min negative={neg_under_setpoint:.2f}°")
        if sign_changes.size > 0:
            print(f"  SCENARIO: Overshoot (Setpoint Crossed) -> result is POSITIVE")
        else:
            print(f"  SCENARIO: Undershoot (Setpoint Not Reached) -> result is NEGATIVE")
        print(f"  Raw peak value: {peak_value:+.2f}°, Final reported value: {signed_over_setpoint:+.2f}°")

    return S, {
        'pos_over_setpoint': pos_over_setpoint,
        'neg_under_setpoint': neg_under_setpoint,
        'signed_over_setpoint': signed_over_setpoint,
        'pos_over_final': pos_over_final,
        'neg_under_final': neg_under_final,
        'signed_over_final': signed_over_final,
        'y_ss': y_ss
    }



def evaluate_full_metrics(
    model,
    env,
    n_episodes=5,
    target_value=0.0,
    tolerance=0.1,
    stable_duration=0.1,
    sim_timestep=0.01,
    live_plot=False,
    save_plots=True,
    output_dir="plots",
    base_seed = 42,
):
    """ Evaluate a policy across episodes and compute rich control metrics.

        Runs `n_episodes` rollouts, logs trajectories, computes stabilization
        (time-to-band with a **dwell** requirement), several steady-state
        statistics, integral metrics, overshoot/undershoot in **degrees**, and
        (optionally) saves per-episode and summary plots plus a text log.

        Parameters
        ----------
        model
            A Stable-Baselines3 policy with `.predict(obs, deterministic=True)`.
        env
            Gym/Gymnasium-compatible environment (e.g., `SimulinkEnv`).
        n_episodes : int, optional
            Number of episodes to evaluate.
        target_value : float, optional
            Target angle in **radians** (commonly 0.0).
        tolerance : float, optional
            Stabilization band half-width in **radians**.
        stable_duration : float, optional
            Required continuous time (seconds) inside the tolerance band to
            declare stabilization (dwell time).
        sim_timestep : float, optional
            Simulation timestep used for time accumulation (seconds).
        live_plot : bool, optional
            If True, show a live Matplotlib plot during each episode.
        save_plots : bool, optional
            If True, save per-episode plots, a combined subplot, a theta-only
            plot with tolerance bands, and a results text log.
        output_dir : str, optional
            Directory for figures and logs (created if missing).
        base_seed : int, optional
            Base seed; episode `i` uses `base_seed + i` when supported by `env`.

        Returns
        -------
        dict
            Dictionary of episode-level arrays and summary scalars:
            - `"rewards"` : list[float]
            - `"stabilisation_times"` : list[float] (seconds; with dwell)
            - `"steady_state_errors"` : list[float] (deg; **MAE** vs target)
            - `"overshoots"` : list[float] (deg; signed over/under-setpoint)
            - `"total_stable_times"` : list[float] (seconds inside band)
            - `"percent_of_max"` : float (mean reward as % of theoretical max)
            - `"steady_rmse"` : list[float] (deg; steady-state RMSE)
            - `"steady_max"` : list[float] (deg; max |steady-state error|)
            - `"osc_std"` : list[float] (deg; std. dev. of steady-state theta)
            - `"iae"` : list[float] (deg·s; integral of |error|)
            - `"u_energy"` : list[float] (∫u² dt; control effort proxy)

        Side Effects
        ------------
        - When `save_plots=True`, writes PNGs and a results TXT file under
        `output_dir/<algo>/<noise_tag>/<run_name>/...`.

        Notes
        -----
        - Compatible with both Gymnasium `(obs, info)` reset and legacy Gym.
        - Stabilization time is the first index for which the in-band dwell
        persists for `stable_duration`.
        - Many metrics are computed in **degrees** for interpretability; inputs
        and control evolve in native units (rad, N) within the env.
    """

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    log_lines = []

    all_rewards = []
    steady_offset_list = []
    stabilisation_times = []
    steady_state_errors = []
    total_stable_times = []
    all_thetas = []
    all_theta_vs = []
    all_us = []
    overshoots = []
    steady_rmse_list = []
    steady_max_list = []
    osc_std_list = []
    iae_list = []
    u_energy_list = []

    for ep in range(n_episodes):

        seed_val = int(base_seed + ep) if base_seed is not None else None
        if seed_val is not None:
            try:
                res = env.reset(seed=seed_val)         
            except TypeError:
                if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "seed"):
                    env.unwrapped.seed(seed_val)
                elif hasattr(env, "seed"):
                    env.seed(seed_val)
                res = env.reset()
        else:
            res = env.reset()
        obs = res[0] if isinstance(res, tuple) else res 

        done = False
        t = 0.0
        ep_reward = 0.0
        total_stable = 0.0
        episode_thetas, episode_theta_vs, episode_us, t_vals = [], [], [], []
        in_band_run = 0.0
        stabilised_idx = None


        # live plotting setup
        if live_plot:
            plt.ion()
            fig_live, axs_live = plt.subplots(4, 1, figsize=(8, 10))
            (theta_line,) = axs_live[0].plot([], [], label="theta (deg)")
            (theta_v_line,) = axs_live[1].plot([], [], label="theta_v (deg/s)")
            (u_line,) = axs_live[2].plot([], [], label="u (N)")
            for ax, title, ylabel in zip(
                axs_live[:3],
                ["Theta (deg)", "Theta Velocity (deg/s)", "Control Input"],
                ["°", "°/s", "N"],
            ):
                ax.set(title=title, xlabel="Time (s)", ylabel=ylabel)
                ax.grid()
                ax.legend()
            axp = axs_live[3]
            axp.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal")
            (pend_line,) = axp.plot([], [], lw=4)

        # episode loop
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            out = env.step(action)
            if isinstance(out, tuple) and len(out) == 5:
                obs, reward, terminated, truncated, _ = out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, _ = out

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
                    ax.relim()
                    ax.autoscale_view()
                pend_line.set_data([0, np.sin(theta)], [0, -np.cos(theta)])
                axs_live[3].set_title(f"Angle: {theta_deg:.1f}°")
                plt.pause(0.001)

            err = abs(theta - target_value)
            if err < tolerance:
                total_stable += sim_timestep
                in_band_run += sim_timestep
                if stabilised_idx is None and in_band_run >= stable_duration:
                    stabilised_idx = len(t_vals) - 1
            else:
                in_band_run = 0.0

            t += sim_timestep

        if live_plot:
            plt.ioff()
            plt.close(fig_live)

        stab_time = t_vals[stabilised_idx] if stabilised_idx is not None else t

        tail_len = max(20, int(0.15 * len(episode_thetas)))
        if stabilised_idx is not None:
            ss_start_idx = stabilised_idx
        else:
            ss_start_idx = max(0, len(episode_thetas) - tail_len)
        
        ss_thetas = np.array(episode_thetas[ss_start_idx:])
        ss_err_deg = ss_thetas - np.degrees(target_value)
        steady_offset = np.mean(ss_err_deg)
        steady_mae = float(np.mean(np.abs(ss_err_deg)))
        steady_rmse = float(np.sqrt(np.mean(ss_err_deg**2)))
        steady_max = float(np.max(np.abs(ss_err_deg)))
        osc_std = float(np.std(ss_thetas))

        all_e_deg = np.array(episode_thetas) - np.degrees(target_value)
        iae = float(np.trapz(np.abs(all_e_deg), t_vals))
        u_energy = float(np.sum(np.square(episode_us)) * sim_timestep)

        # Keep existing overshoot calculation
        setpoint_deg = np.degrees(target_value)
        theta_arr = np.asarray(episode_thetas, dtype=float)
        t_arr = np.asarray(t_vals, dtype=float)
        S, over_dict = signed_over_from_stepinfo(theta_arr, t_arr, setpoint_deg)
        over = over_dict['signed_over_setpoint']
        
        all_rewards.append(ep_reward)
        stabilisation_times.append(stab_time)
        steady_offset_list.append(steady_offset)
        steady_state_errors.append(steady_mae)
        total_stable_times.append(total_stable)
        all_thetas.append((t_vals, episode_thetas))
        all_theta_vs.append((t_vals, episode_theta_vs))
        all_us.append((t_vals, episode_us))
        overshoots.append(over)
        steady_rmse_list.append(steady_rmse)
        steady_max_list.append(steady_max)
        osc_std_list.append(osc_std)
        iae_list.append(iae)
        u_energy_list.append(u_energy)

        lines = [
            f"Episode {ep+1}:",
            f"  Total reward                : {ep_reward:.2f}",
            f"  Stabilisation time          : {stab_time:.2f} s",
            f"  Steady-state offset         : {steady_offset:+.2f}°",
            f"  Overshoot(+)/Undershoot(-)  : {over:+.2f}°",
            f"  Total stable time           : {total_stable:.2f} s",
            f"  Steady-state MAE            : {steady_mae:.2f}°",
            f"  Steady-state RMSE           : {steady_rmse:.2f}°",
            f"  Steady-state max error      : {steady_max:.2f}°",
            f"  Oscillation (σ)             : {osc_std:.2f}°",
            f"  IAE (|deg|·s)               : {iae:.2f}",
            f"  Control energy              : {u_energy:.2f}",
        ]
        for l in lines:
            print(l)
            log_lines.append(l)

        # save per-episode plot
        if save_plots:
            fname = os.path.join(
                output_dir, f"{model.__class__.__name__}_episode_{ep+1}.png"
            )
            fig_ep, ax_ep = plt.subplots(figsize=(6, 3))
            ax_ep.plot(t_vals, episode_thetas, label=f"Episode {ep+1}")
            tol_deg = np.degrees(tolerance)
            ax_ep.axhline(
                tol_deg, linestyle="--", color="gray", label=f"+{tol_deg:.1f}° tol"
            )
            ax_ep.axhline(
                -tol_deg, linestyle="--", color="gray", label=f"-{tol_deg:.1f}° tol"
            )
            ax_ep.axhline(0, linestyle="--", color="red")
            ax_ep.set(
                title=f"Theta Trajectory - Episode {ep+1}",
                xlabel="Time (s)",
                ylabel="Theta (deg)",
            )
            ax_ep.grid()
            ax_ep.legend(loc="lower right")
            fig_ep.tight_layout()
            fig_ep.savefig(fname, bbox_inches="tight")
            plt.close(fig_ep)

    max_r = env.unwrapped.max_episode_time / sim_timestep
    mean_r, std_r = np.mean(all_rewards), np.std(all_rewards)
    percent_max = 100 * mean_r / max_r
    summary_lines = [
        "",
        "Summary Metrics:",
        "-" * 40,
        f"Mean Reward: {mean_r:.2f} ± {std_r:.2f} (Max: {max_r:.0f})",
        f"→ {percent_max:.1f}% of theoretical max reward",
        f"Mean Stabilisation Time             : {np.mean(stabilisation_times):.2f} s",
        f"Mean Steady-state offset            : {np.mean(steady_offset_list):.2f}°",
        f"Mean Total Stable Time              : {np.mean(total_stable_times):.2f} s",
        f"Mean Steady-State MAE               : {np.mean(steady_state_errors):.2f}°",
        f"Mean Steady-State RMSE              : {np.mean(steady_rmse_list):.2f}°",
        f"Mean Oscillation (σ)                : {np.mean(osc_std_list):.2f}°",
        f"Mean |Overshoot/Undershoot|         : {np.mean(np.abs(overshoots)):.2f}°",
        f"Mean IAE                            : {np.mean(iae_list):.2f}",
        f"Mean Control Energy                 : {np.mean(u_energy_list):.2f}",
    ]
    for l in summary_lines:
        print(l)
        log_lines.append(l)

    # save final plots and logs
    if save_plots:
        # combined subplot (legends on right; full degree labels)
        fig_all, axs_all = plt.subplots(3, 1, figsize=(8, 10))
        labels = [f"Episode {i+1}" for i in range(len(all_thetas))]
        for idx, ((t_v, th), (_, tv), (_, u)) in enumerate(
            zip(all_thetas, all_theta_vs, all_us)
        ):
            axs_all[0].plot(t_v, th, label=labels[idx])
            axs_all[1].plot(t_v, tv, label=labels[idx])
            axs_all[2].plot(t_v, u, label=labels[idx])
        titles = ["Theta (deg)", "Theta Velocity (deg/s)", "Control Input (N)"]
        ylabels = ["Theta (deg)", "Theta Velocity (deg/s)", "Control Input (N)"]
        for i, ax in enumerate(axs_all):
            ax.set(title=titles[i], xlabel="Time (s)", ylabel=ylabels[i])
            ax.grid(True)
            loc = "upper right" if i < 2 else "lower right"
            ax.legend(loc=loc)
        fig_all.tight_layout()
        fig_all.savefig(
            os.path.join(output_dir, "combined_subplot.png"), bbox_inches="tight"
        )
        plt.close(fig_all)

        tol_deg = np.degrees(tolerance)
        fig_th, ax_th = plt.subplots(figsize=(8, 4))
        for idx, (t_v, th) in enumerate(all_thetas):
            ax_th.plot(t_v, th, label=labels[idx])
        ax_th.axhline(0, linestyle="--", color="red", label="0°")
        ax_th.axhline(
            tol_deg, linestyle="--", color="gray", label=f"+{tol_deg:.1f}° tol"
        )
        ax_th.axhline(
            -tol_deg, linestyle="--", color="gray", label=f"-{tol_deg:.1f}° tol"
        )
        ax_th.set(
            title="All Episodes: Theta (deg)", xlabel="Time (s)", ylabel="Theta (deg)"
        )
        ax_th.grid(True)
        ax_th.legend(loc="upper right")
        fig_th.tight_layout()
        fig_th.savefig(os.path.join(output_dir, "theta_plot.png"), bbox_inches="tight")
        plt.close(fig_th)

        log_file = os.path.join(output_dir, f"{model.__class__.__name__}_results.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

    return {
        "rewards": all_rewards,
        "stabilisation_times": stabilisation_times,
        "steady_state_errors": steady_state_errors,
        "overshoots": overshoots,
        "total_stable_times": total_stable_times,
        "percent_of_max": percent_max,
        "steady_rmse": steady_rmse_list,
        "steady_max": steady_max_list,
        "osc_std": osc_std_list,
        "iae": iae_list,
        "u_energy": u_energy_list
    }


ALGOS = {"DQN": DQN, "DDPG": DDPG, "PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3}

def infer_algo_from_name(folder_name: str) -> str | None:
    """Infer the algorithm key from a folder name using known tokens.

        Performs a case-insensitive substring check against the known algorithm
        keys in `ALGOS` (e.g., 'DQN', 'DDPG', 'PPO', 'A2C', 'SAC', 'TD3').

        Parameters
        ----------
        folder_name : str
            Folder name to inspect (e.g., a run directory name).

        Returns
        -------
        str | None
            Matching algorithm key if found, otherwise `None`.
    """

    u = folder_name.upper()
    for k in ALGOS.keys():
        if k in u:
            return k
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,  # e.g., dqn, ddpg, ppo, a2c, sac, td3, or all
                        help="Algorithm to eval (dqn, ddpg, ppo, a2c, sac, td3, or all)")
    parser.add_argument("--root", default=os.path.join("CP_JAX", "jax_clean"),
                        help="Folder with per-run subfolders (default: ip_jax/jax_models)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-save-plots", dest="no_save_plots", action="store_true")
    parser.add_argument("--env-noise", type=float, default=0.0,
                    help="Evaluation-time observation noise sigma (radians)")

    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"Root not found: {args.root}")

    request = args.algo.upper()
    if request != "ALL" and request not in ALGOS:
        raise SystemExit(f"Unknown algo: {args.algo}. Choose one of {sorted(ALGOS) + ['ALL']}")

    # Discover runs
    runs = []
    for name in sorted(os.listdir(args.root)):
        sub = os.path.join(args.root, name)
        if not os.path.isdir(sub):
            continue
        algo_key = infer_algo_from_name(name)
        if request != "ALL":
            # only keep folders matching the requested algo
            if algo_key != request:
                continue
        else:
            if algo_key is None:
                continue

        zip_path = os.path.join(sub, "best_model.zip")
        if not os.path.isfile(zip_path):
            print(f"[skip] {name}: best_model.zip not found")
            continue
        runs.append((algo_key, name, zip_path))

    if not runs:
        raise SystemExit(f"No runs found for '{args.algo}' under: {args.root}")

    results = {}
    for algo_key, run_name, zip_path in runs:
        print(f"\n=== Evaluating {run_name} (env_noise={args.env_noise:.3f}) ===")
        env = SimulinkEnv(eval_obs_noise_std=args.env_noise)


        # DQN needs discrete action mapping
        if algo_key == "DQN":
            max_torque = float(env.action_space.high[0])
            torque_values = np.linspace(-max_torque, max_torque, 5)
            env = DiscretizedActionWrapper(env, force_values=torque_values)

        ModelClass = ALGOS[algo_key]
        model = (ModelClass.load(zip_path, device="cpu")
                 if algo_key in ("PPO", "A2C") else ModelClass.load(zip_path))

        results[f"{algo_key}/{run_name}"] = evaluate_full_metrics(
            model,
            env,
            n_episodes=args.episodes,
            live_plot=False,
            save_plots=not args.no_save_plots,
            output_dir=os.path.join(
                "plots",
                algo_key,
                f"{algo_key.lower()}_env_noise_{args.env_noise:.3f}",
                run_name,
            ),
            base_seed=42,                    
        )


    print("\nAll evaluations complete.")