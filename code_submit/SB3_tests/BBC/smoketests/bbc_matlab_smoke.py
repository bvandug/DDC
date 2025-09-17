import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from BBCSimulink_env import BBCSimulinkEnv


def duty_from_pattern(pattern: str, t: float, clamp=(0.1, 0.9), **kw) -> float:
    """Generate a duty command at time `t` from a named pattern.

        Supported patterns and kwargs:
        - "sine":     freq (Hz), amp (0..1), offset (0..1)
        - "step":     step_high (0..1), step_low (0..1), t_switch (s)
        - "const":    const_val (0..1)
        - "ramp":     ramp_start (0..1), ramp_end (0..1), ramp_period (s)

        The resulting value is clipped to `clamp=(lo, hi)`.

        Parameters
        ----------
        pattern : str
            One of {"sine", "step", "const", "ramp"}.
        t : float
            Current time (seconds).
        clamp : tuple[float, float], optional
            Lower/upper bounds for the returned duty (default (0.1, 0.9)).
        **kw :
            Pattern-specific keyword arguments (see above).

        Returns
        -------
        float
            Duty cycle in [clamp[0], clamp[1]].

        Raises
        ------
        ValueError
            If an unknown pattern is provided.
    """

    lo, hi = clamp
    if pattern == "sine":
        f = float(kw.get("freq", 5.0))
        amp = float(kw.get("amp", 0.4))
        offset = float(kw.get("offset", 0.5))
        val = offset + amp * np.sin(2.0 * np.pi * f * t)
    elif pattern == "step":
        high = float(kw.get("step_high", 0.9))
        low = float(kw.get("step_low", 0.1))
        t_switch = float(kw.get("t_switch", 0.05))
        val = high if t < t_switch else low
    elif pattern == "const":
        val = float(kw.get("const_val", 0.5))
    elif pattern == "ramp":
        start = float(kw.get("ramp_start", 0.1))
        end = float(kw.get("ramp_end", 0.9))
        period = float(kw.get("ramp_period", 0.1))
        if period <= 0:
            period = 0.1
        phase = (t % period) / period
        val = start + (end - start) * phase
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return float(np.clip(val, lo, hi))


def run_smoke_episode(env: BBCSimulinkEnv, episode_idx: int, outdir: str,
                      pattern: str, live_plot: bool, **pattern_kw):
    """ Run one smoke-test episode using a synthetic duty pattern.

        Rolls the environment forward by one switching period per step using the
        selected pattern, logs time, capacitor voltage, duty, and (if available)
        inductor current, optionally updates live Matplotlib plots, then calls
        `finalize` to save plots and data.

        Parameters
        ----------
        env : BBCSimulinkEnv
            The instantiated Simulink-backed buck–boost environment.
        episode_idx : int
            1-based episode index used in filenames.
        outdir : str
            Output directory for plots and saved data (created if missing).
        pattern : str
            Duty pattern name passed to `duty_from_pattern`.
        live_plot : bool
            If True, update live plots during the episode.
        **pattern_kw :
            Keyword args forwarded to `duty_from_pattern` (e.g., freq, amp, etc.).

        Returns
        -------
        dict
            The result dict returned by `finalize`, including file paths, reward,
            and step count.
    """
    os.makedirs(outdir, exist_ok=True)

    obs, info = env.reset()
    target = float(obs[3])
    T_sw = float(info.get("T_sw", env.T_sw))

    t_list, vC_list, duty_list, iL_list = [], [], [], []
    total_reward = 0.0

    if live_plot:
        plt.ion()
        fig1 = plt.figure(figsize=(10, 4)); ax1 = fig1.add_subplot(111)
        line_vc, = ax1.plot([], [], label="vC (V)")
        line_vref, = ax1.plot([], [], linestyle=":", label="target (V)")
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Voltage (V)"); ax1.legend()
        fig2 = plt.figure(figsize=(10, 3)); ax2 = fig2.add_subplot(111)
        line_duty, = ax2.plot([], [], label="duty")
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Duty (0..1)"); ax2.set_ylim(0, 1)

    t = 0.0
    while True:
        duty_val = duty_from_pattern(pattern, t, **pattern_kw)
        action = np.array([duty_val], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        vC = float(info.get("vC", obs[0]))
        duty = float(info.get("duty_cmd", action[0]))
        iL = info.get("iL", None)

        t_list.append(t)
        vC_list.append(vC)
        duty_list.append(duty)
        iL_list.append(np.nan if iL is None else float(iL))

        if live_plot:
            line_vc.set_data(t_list, vC_list)
            line_vref.set_data(t_list, [target] * len(t_list))
            ax1.relim(); ax1.autoscale_view()
            line_duty.set_data(t_list, duty_list)
            ax2.relim(); ax2.autoscale_view(); ax2.set_ylim(0, 1)
            plt.pause(0.001)

        t += T_sw
        if terminated or truncated:
            break

    return finalize(outdir, episode_idx, np.asarray(t_list), np.asarray(vC_list),
                    np.asarray(duty_list), np.asarray(iL_list), target, total_reward)


def finalize(outdir, episode_idx, t_arr, vC_arr, duty_arr, iL_arr, target, total_reward):
    """ Save plots and data for an episode, and return file paths/metrics.

        Creates and saves:
        - Voltage vs. time plot (with target overlay)
        - Duty vs. time plot
        - Optional inductor current vs. time plot (if any non-NaN values)
        - Compressed `.npz` with arrays (t, vC, duty, iL, target, total_reward)
        - CSV with columns: t, vC, duty, iL

        Parameters
        ----------
        outdir : str
            Destination folder for all artifacts.
        episode_idx : int
            1-based episode index used in filenames.
        t_arr : np.ndarray
            Time vector (seconds).
        vC_arr : np.ndarray
            Capacitor voltage trace (volts).
        duty_arr : np.ndarray
            Duty command trace (0..1).
        iL_arr : np.ndarray
            Inductor current trace (amps); may contain NaNs.
        target : float
            Target voltage (volts).
        total_reward : float
            Accumulated reward over the episode.

        Returns
        -------
        dict
            {
            "plots": {"voltage": str, "duty": str, "iL": str | None},
            "npz": str,
            "csv": str,
            "reward": float,
            "steps": int,
            }
    """

    # Voltage plot
    fig_v = plt.figure(figsize=(10, 4)); axv = fig_v.add_subplot(111)
    axv.plot(t_arr, vC_arr, label="vC (V)")
    axv.plot(t_arr, np.full_like(t_arr, target), linestyle=":", label="target (V)")
    axv.set_title(f"Smoke Test — Episode {episode_idx} (reward={total_reward:.2f})")
    axv.set_xlabel("Time (s)"); axv.set_ylabel("Voltage (V)"); axv.legend(loc="best")
    fig_v.tight_layout()
    v_path = os.path.join(outdir, f"ep{episode_idx:02d}_voltage.png"); fig_v.savefig(v_path, dpi=160)

    # Duty plot
    fig_d = plt.figure(figsize=(10, 3.2)); axd = fig_d.add_subplot(111)
    axd.plot(t_arr, duty_arr, label="duty")
    axd.set_xlabel("Time (s)"); axd.set_ylabel("Duty (0..1)"); axd.set_ylim(0.0, 1.0); axd.legend(loc="best")
    fig_d.tight_layout()
    d_path = os.path.join(outdir, f"ep{episode_idx:02d}_duty.png"); fig_d.savefig(d_path, dpi=160)

    i_path = None
    if not np.all(np.isnan(iL_arr)):
        fig_i = plt.figure(figsize=(10, 3.2)); axi = fig_i.add_subplot(111)
        axi.plot(t_arr, iL_arr, label="iL (A)")
        axi.set_xlabel("Time (s)"); axi.set_ylabel("Inductor Current (A)"); axi.legend(loc="best")
        fig_i.tight_layout()
        i_path = os.path.join(outdir, f"ep{episode_idx:02d}_iL.png"); fig_i.savefig(i_path, dpi=160)

    npz_path = os.path.join(outdir, f"ep{episode_idx:02d}.npz")
    np.savez_compressed(npz_path, t=t_arr, vC=vC_arr, duty=duty_arr, iL=iL_arr,
                        target=np.array([target], dtype=float),
                        total_reward=np.array([total_reward], dtype=float))
    
    csv_path = os.path.join(outdir, f"ep{episode_idx:02d}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t,vC,duty,iL\n")
        for tt, vv, dd, ii in zip(t_arr, vC_arr, duty_arr, iL_arr):
            ii_str = "" if np.isnan(ii) else f"{ii:.9g}"
            f.write(f"{tt:.9g},{vv:.9g},{dd:.9g},{ii_str}\n")

    return {
        "plots": {"voltage": v_path, "duty": d_path, "iL": i_path},
        "npz": npz_path,
        "csv": csv_path,
        "reward": float(total_reward),
        "steps": int(len(t_arr)),
    }


def main():
    """ CLI entry point: run smoke tests on BBCSimulinkEnv with duty patterns.

        Parses command-line options, constructs `BBCSimulinkEnv`, runs the
        requested number of episodes with the specified pattern (sine/step/const/
        ramp), saves per-episode artifacts via `run_smoke_episode`, generates an
        overlay plot of vC across episodes, prints a short summary, and closes
        the environment.

        Returns
        -------
        None
    """

    p = argparse.ArgumentParser(description="Smoke test for BBCSimulinkEnv with synthetic duty patterns")
    p.add_argument("--outdir", type=str, default="plots_bbc_smoke")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--live-plot", action="store_true")

    p.add_argument("--model-name", type=str, default="bbcSim")
    p.add_argument("--dt", type=float, default=5e-6)
    p.add_argument("--frame-skip", type=int, default=10)
    p.add_argument("--episode-time", type=float, default=0.2)
    p.add_argument("--grace-steps", type=int, default=100)
    p.add_argument("--target", type=float, default=-30.0)
    p.add_argument("--random-target", action="store_true")

    p.add_argument("--pattern", type=str, default="sine",
                   choices=["sine", "step", "const", "ramp"])

    p.add_argument("--freq", type=float, default=5.0, help="sine frequency (Hz)")
    p.add_argument("--amp", type=float, default=0.4, help="sine amplitude around offset")
    p.add_argument("--offset", type=float, default=0.5, help="sine center (duty)")

    p.add_argument("--step-high", type=float, default=0.9)
    p.add_argument("--step-low", type=float, default=0.1)
    p.add_argument("--t-switch", type=float, default=0.05, help="seconds until switch low")

    p.add_argument("--const", dest="const_val", type=float, default=0.5)

    p.add_argument("--ramp-start", type=float, default=0.1)
    p.add_argument("--ramp-end", type=float, default=0.9)
    p.add_argument("--ramp-period", type=float, default=0.1)

    args = p.parse_args()

    env = BBCSimulinkEnv(
        model_name=args.model_name,
        dt=args.dt,
        frame_skip=args.frame_skip,
        max_episode_time=args.episode_time,
        grace_period_steps=args.grace_steps,
        target_voltage=args.target,
        random_target=bool(args.random_target),
        enable_plotting=False,
        use_fast_restart=True,
    )

    print(f"Running smoke test: pattern={args.pattern}")
    summary = []
    for ep in range(1, args.episodes + 1):
        res = run_smoke_episode(
            env=env,
            episode_idx=ep,
            outdir=args.outdir,
            pattern=args.pattern,
            live_plot=args.live_plot,
            freq=args.freq, amp=args.amp, offset=args.offset,
            step_high=args.step_high, step_low=args.step_low, t_switch=args.t_switch,
            const_val=args.const_val,
            ramp_start=args.ramp_start, ramp_end=args.ramp_end, ramp_period=args.ramp_period,
        )
        summary.append((ep, res["reward"], res["steps"]))
        print(f"Episode {ep}: reward={res['reward']:.2f}, steps={res['steps']}")

    fig_sum = plt.figure(figsize=(11, 5)); ax_sum = fig_sum.add_subplot(111)
    last_t = None; target = args.target
    for ep in range(1, args.episodes + 1):
        data = np.load(os.path.join(args.outdir, f"ep{ep:02d}.npz"))
        ax_sum.plot(data["t"], data["vC"], label=f"ep {ep}")
        last_t = data["t"]
        target = float(data["target"][0])
    if last_t is not None:
        ax_sum.plot(last_t, np.full_like(last_t, target), linestyle=":", label="target")
    ax_sum.set_title("Smoke Test — vC across episodes")
    ax_sum.set_xlabel("Time (s)"); ax_sum.set_ylabel("Voltage (V)")
    ax_sum.legend(loc="best", ncol=2)
    fig_sum.tight_layout(); fig_sum.savefig(os.path.join(args.outdir, "summary_voltage.png"), dpi=160)

    print("Summary:")
    for ep, rew, steps in summary:
        print(f"  ep {ep:02d}: reward={rew:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    main()
