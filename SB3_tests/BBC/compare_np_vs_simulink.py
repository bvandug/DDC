#!/usr/bin/env python3
"""
compare_np_vs_simulink.py
---------------------------------
Feed identical duty sequences to the NumPy Buck-Boost env (np_bbc_env) and the
Simulink-backed env (BBCSimulink_env), then compare trajectories and plot.

Usage examples
--------------
# Constant duty 0.38 for 2000 PWM periods (50 µs @ 20 kHz)
python compare_np_vs_simulink.py --mode constant --constant-duty 0.38 --steps 2000

# Sine duty around 0.38 ± 0.05 at 40 Hz for 4000 periods
python compare_np_vs_simulink.py --mode sine --center 0.38 --amp 0.05 --freq-hz 40 --steps 4000

# Single step duty: 0.2 for 1s then 0.5 (use --step-at to set change time)
python compare_np_vs_simulink.py --mode step --duty-a 0.2 --duty-b 0.5 --step-at 0.02 --steps 20000

Notes
-----
- One RL/env step maps to exactly one PWM period (frame_skip * dt).
- We keep dt/frame_skip/target identical between envs.
- CSV columns contain time (s), duty, vC_np, vC_sim, iL_np, iL_sim (if available), error terms.
- Plots: duty, vC overlay, iL overlay (if available), and vC difference (sim - np).
- Plots now include inline annotations: RMSE/MAE text boxes, steady-state window shading,
  and ±2% target band shading on the vC overlay.
"""

import argparse
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

from np_bbc_env import JAXBuckBoostConverterEnv   # NumPy env
from BBCSimulink_env import BBCSimulinkEnv        # Simulink env

def duty_sequence(mode: str, steps: int, dt: float, frame_skip: int,
                  center: float, amp: float, freq_hz: float,
                  constant_duty: float, duty_a: float, duty_b: float, step_at: float):
    """Generate a duty sequence of length `steps`."""
    Tsw = dt * frame_skip
    t = np.arange(steps, dtype=float) * Tsw
    if mode == "constant":
        d = np.full(steps, constant_duty, dtype=float)
    elif mode == "sine":
        # center +/- amp, clipped to [0.1, 0.9]
        d = center + amp * np.sin(2*np.pi*freq_hz*t)
        d = np.clip(d, 0.1, 0.9)
    elif mode == "step":
        d = np.full(steps, duty_a, dtype=float)
        idx = int(round(step_at / Tsw))
        idx = max(0, min(steps, idx))
        d[idx:] = duty_b
        d = np.clip(d, 0.1, 0.9)
    else:
        raise ValueError(f"Unknown mode {mode}")
    return t, d

def run_one(env_np, env_sim, duties, sim_extra_steps=1, sim_drop_last=1):
    import numpy as np
    steps = len(duties)
    obs_np, _ = env_np.reset(seed=123)
    obs_sm, _ = env_sim.reset(seed=123)
    Tsw = env_np.dt * env_np.frame_skip

    # ---- NumPy (exactly 'steps') ----
    rec_np = {"t": np.arange(steps)*Tsw, "duty": duties.copy(),
              "vC": np.zeros(steps), "iL": np.zeros(steps),
              "err": np.zeros(steps), "r": np.zeros(steps),
              "term": np.zeros(steps, bool), "trunc": np.zeros(steps, bool)}
    for k, d in enumerate(duties):
        o, r, te, tr, info = env_np.step(np.array([d], np.float32))
        rec_np["vC"][k] = float(info.get("vC", o[0]))
        rec_np["iL"][k] = float(info.get("iL", np.nan))
        rec_np["err"][k] = float(info.get("err", o[1]))
        rec_np["r"][k] = float(r); rec_np["term"][k] = bool(te); rec_np["trunc"][k] = bool(tr)
        if te or tr:  # trim if NP ends early
            for key in rec_np: rec_np[key] = rec_np[key][:k+1]
            steps = k+1
            break

    # ---- Simulink (steps + extra), then drop tail ----
    sim_steps = steps + max(0, int(sim_extra_steps))
    sim_duties = np.pad(duties[:steps], (0, sim_steps-steps), mode="edge")
    rec_sm = {"t": np.arange(sim_steps)*Tsw, "duty": sim_duties.copy(),
              "vC": np.zeros(sim_steps), "iL": np.full(sim_steps, np.nan),
              "err": np.zeros(sim_steps), "r": np.zeros(sim_steps),
              "term": np.zeros(sim_steps, bool), "trunc": np.zeros(sim_steps, bool),
              "gate_meas": [ ], "duty_meas_sim": np.full(sim_steps, np.nan)}  # per-step measured duty (if provided)
             
    for k, d in enumerate(sim_duties):
        o, r, te, tr, info = env_sim.step(np.array([d], np.float32))

        # --- measured duty heuristics (use whatever Simulink returns) ---
        gfrac = None
        if "gate_frac" in info:
            # directly provided per-period high-time fraction
            gfrac = float(info["gate_frac"])
        elif "gate" in info:
            # timeseries of gate within this period -> compute fraction of time high
            g = np.asarray(info["gate"]).astype(float).ravel()
            if g.size > 0:
                # if it's logical 0/1 samples
                gfrac = float(np.mean(g > 0.5))
        elif "duty_meas" in info:
            gfrac = float(info["duty_meas"])

        if gfrac is not None:
            rec_sm["duty_meas_sim"][k] = gfrac

        rec_sm["vC"][k] = float(info.get("vC", o[0]))
        iL = info.get("iL", None); rec_sm["iL"][k] = (np.nan if iL is None else float(iL))
        rec_sm["err"][k] = float(info.get("err", o[1]))
        rec_sm["r"][k] = float(r); rec_sm["term"][k] = bool(te); rec_sm["trunc"][k] = bool(tr)
        if te:
            for key in rec_sm: rec_sm[key] = rec_sm[key][:k+1]
            sim_steps = k+1; break
        
        # ⬇️ NEW: capture and print measured duty if Simulink provided it
        md = info.get("measured_duty", None)
        if md is not None:
            # store it so we can summarize/save later
            rec_sm["duty_meas_sim"][k] = float(md)
            # lightweight debug prints (first 10, then every 200th, and the last)
            if k < 10 or (k % 200 == 0) or te:
                print(f"[sim] k={k:5d}  duty_cmd={d:.4f}  measured_duty={md:.4f}")

    drop = max(0, int(sim_drop_last))
    if drop and sim_steps > drop:
        for key in rec_sm:
            rec_sm[key] = rec_sm[key][:-drop]
        sim_steps -= drop


    # ---- Align and return ----
    n = min(steps, sim_steps)
    return {
        "t": rec_np["t"][:n], "duty": rec_np["duty"][:n],
        "vC_np": rec_np["vC"][:n], "vC_sim": rec_sm["vC"][:n],
        "iL_np": rec_np["iL"][:n], "iL_sim": rec_sm["iL"][:n],
        "err_np": rec_np["err"][:n], "err_sim": rec_sm["err"][:n],
        "r_np": rec_np["r"][:n], "r_sim": rec_sm["r"][:n],
        "term_np": rec_np["term"][:n], "term_sim": rec_sm["term"][:n],
        "trunc_np": rec_np["trunc"][:n], "trunc_sim": rec_sm["trunc"][:n],
        "duty_meas_sim": rec_sm["duty_meas_sim"][:n],

    }

def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n is None or n <= 1 or x.size < n:
        return x.copy()
    w = np.ones(n, dtype=float) / n
    return np.convolve(x, w, mode="valid")

def summarize(rec, target_voltage, ma_window=0):
    v_np = rec["vC_np"]; v_sm = rec["vC_sim"]
    n = len(v_np)
    if n < 2:
        return {}

    # raw metrics
    diff = v_sm - v_np
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae  = float(np.mean(np.abs(diff)))
    tail = max(50, n // 10)
    v_np_ss = float(np.mean(v_np[-tail:]))
    v_sm_ss = float(np.mean(v_sm[-tail:]))
    ss_err_np = float(abs(v_np_ss - target_voltage))
    ss_err_sm = float(abs(v_sm_ss - target_voltage))
    ss_delta  = float(v_sm_ss - v_np_ss)

    stats = {
        "steps": n,
        "rmse_vC": rmse,
        "mae_vC": mae,
        "v_np_ss": v_np_ss,
        "v_sm_ss": v_sm_ss,
        "ss_err_np": ss_err_np,
        "ss_err_sim": ss_err_sm,
        "ss_delta_sim_minus_np": ss_delta,
        "ss_tail_len": tail,
    }

    # moving-average (period-averaged over multiple periods)
    if ma_window and ma_window > 1 and n >= ma_window:
        v_np_ma = _moving_average(v_np, ma_window)
        v_sm_ma = _moving_average(v_sm, ma_window)
        diff_ma = v_sm_ma - v_np_ma
        stats.update({
            "ma_window": ma_window,
            "rmse_vC_ma": float(np.sqrt(np.mean(diff_ma**2))),
            "mae_vC_ma":  float(np.mean(np.abs(diff_ma))),
            "v_np_ss_ma": float(np.mean(v_np_ma[-max(5, len(v_np_ma)//10):])),
            "v_sm_ss_ma": float(np.mean(v_sm_ma[-max(5, len(v_sm_ma)//10):])),
        })

    # measured duty (from Simulink) if available
    dm = rec.get("duty_meas_sim", None)
    if dm is not None:
        m = np.asarray(dm)
        mask = np.isfinite(m)
        if np.any(mask):
            stats["measured_duty_sim_mean"] = float(np.mean(m[mask]))
            stats["measured_duty_sim_tail"] = float(np.mean(m[mask][-tail:] if tail < mask.sum() else m[mask]))
            stats["commanded_duty_mean"]    = float(np.mean(rec["duty"]))
    return stats


def _annotate_textbox(ax, lines, loc="upper right"):
    """Add a small annotation box inside axes with given lines of text."""
    text = "\n".join(lines)
    # Choose anchor based on loc
    anchors = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
    }
    x, y, ha, va = anchors.get(loc, anchors["upper right"])
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9),
    )


def save_csv(rec, out_csv: Path):
    import pandas as pd
    df = pd.DataFrame({
        "t_s": rec["t"],
        "duty": rec["duty"],
        "vC_np": rec["vC_np"],
        "vC_sim": rec["vC_sim"],
        "iL_np": rec["iL_np"],
        "iL_sim": rec["iL_sim"],
        "err_np": rec["err_np"],
        "err_sim": rec["err_sim"],
        "r_np": rec["r_np"],
        "r_sim": rec["r_sim"],
        "terminated_np": rec["term_np"],
        "terminated_sim": rec["term_sim"],
        "truncated_np": rec["trunc_np"],
        "truncated_sim": rec["trunc_sim"],
    })
    df.to_csv(out_csv, index=False)
    return df


def make_plots(rec, target_voltage, title_prefix, out_dir: Path, stats: dict):
    import matplotlib.pyplot as plt

    t = rec["t"]
    v_np = rec["vC_np"]
    v_sm = rec["vC_sim"]
    duty = rec["duty"]
    i_np = rec["iL_np"]
    i_sm = rec["iL_sim"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Duty vs time
    plt.figure()
    plt.plot(t, duty, label="Duty")
    plt.axhline(0.1, linestyle="--", label="Min duty (0.1)")
    plt.axhline(0.9, linestyle="--", label="Max duty (0.9)")
    plt.xlabel("Time (s)")
    plt.ylabel("Duty")
    plt.title(f"{title_prefix} — Duty sequence")
    # Legend below to avoid obstruction
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "01_duty.png", dpi=150)
    plt.close()

    # 2) vC overlay + target + ±2% band + steady-state shading + RMSE/MAE
    fig, ax = plt.subplots()
    ax.plot(t, v_np, label="vC (NumPy env)")
    ax.plot(t, v_sm, label="vC (Simulink env)")
    ax.axhline(target_voltage, linestyle="--", label="Target")

    # ±2% band shading
    band = 0.02 * abs(target_voltage)
    ax.axhspan(target_voltage - band, target_voltage + band, alpha=0.15, label="±2% band")

    # Steady-state window shading (last tail samples)
    tail = stats.get("ss_tail_len", max(50, max(1, len(t)//10)))
    if len(t) >= tail:
        t0 = t[-tail]
        ax.axvspan(t0, t[-1], alpha=0.1, label="Steady-state window")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Capacitor voltage vC (V)")
    ax.set_title(f"{title_prefix} — vC overlay")

    # Inline annotation (RMSE/MAE & SS means)
    lines = [
        f"RMSE: {stats.get('rmse_vC', float('nan')):.4f} V",
        f"MAE:  {stats.get('mae_vC', float('nan')):.4f} V",
        f"SS NumPy: {stats.get('v_np_ss', float('nan')):.3f} V",
        f"SS Sim:   {stats.get('v_sm_ss', float('nan')):.3f} V",
        f"ΔSS (Sim-NP): {stats.get('ss_delta_sim_minus_np', float('nan')):.3f} V",
    ]
    _annotate_textbox(ax, lines, loc="upper right")

    # Legend below
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "02_vc_overlay.png", dpi=150)
    plt.close(fig)

    # 3) vC difference (sim - np) + inline stats
    fig, ax = plt.subplots()
    diff = v_sm - v_np
    ax.plot(t, diff, label="vC_sim - vC_np")
    ax.axhline(0.0, linestyle="--", label="Zero")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔvC (V)")
    ax.set_title(f"{title_prefix} — vC difference")

    lines = [
        f"RMSE: {stats.get('rmse_vC', float('nan')):.4f} V",
        f"MAE:  {stats.get('mae_vC', float('nan')):.4f} V",
    ]
    _annotate_textbox(ax, lines, loc="upper right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "03_vc_diff.png", dpi=150)
    plt.close(fig)

    # 4) iL overlay (if Simulink iL available; otherwise still show NumPy iL)
    fig, ax = plt.subplots()
    ax.plot(t, i_np, label="iL (NumPy env)")
    if not np.all(np.isnan(i_sm)):
        ax.plot(t, i_sm, label="iL (Simulink env)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inductor current iL (A)")
    ax.set_title(f"{title_prefix} — iL overlay")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "04_iL_overlay.png", dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    # Shared env parameters
    ap.add_argument("--dt", type=float, default=5e-6, help="Base integration step (s)")
    ap.add_argument("--frame-skip", type=int, default=10, help="Substeps per PWM period (Tsw=dt*frame_skip)")
    ap.add_argument("--target", type=float, default=-30.0, help="Target output voltage (V)")
    ap.add_argument("--grace-steps", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=500, help="Max episode steps (NumPy)")
    ap.add_argument("--max-episode-time", type=float, default=0.2, help="Max episode time (Simulink)")
    ap.add_argument("--model-name", type=str, default="bbcSim", help="Simulink model name")
    ap.add_argument("--enforce-dcm", action="store_true", help="NumPy env: enforce DCM (no negative iL)")
    # Sequence
    ap.add_argument("--mode", choices=["constant","sine","step"], default="constant")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--constant-duty", type=float, default=0.38)
    ap.add_argument("--center", type=float, default=0.38)
    ap.add_argument("--amp", type=float, default=0.05)
    ap.add_argument("--freq-hz", type=float, default=40.0)
    ap.add_argument("--duty-a", type=float, default=0.2)
    ap.add_argument("--duty-b", type=float, default=0.5)
    ap.add_argument("--step-at", type=float, default=0.02, help="Time (s) to switch from duty-a to duty-b")

    ap.add_argument("--sim-extra-steps", type=int, default=0,
                    help="Run Simulink for this many extra steps (uses last duty).")
    ap.add_argument("--sim-drop-last", type=int, default=0,
                    help="After Simulink run, drop this many tail steps before alignment.")
    
    ap.add_argument("--ma-window", type=int, default=25,
                help="Moving-average window (in periods) for DC-bias metrics (0 disables).")

    

    # Output
    ap.add_argument("--out-dir", type=Path, default=Path("np_vs_simulink_out"))
    ap.add_argument("--csv-name", type=str, default="trace.csv")
    args = ap.parse_args()

    # Instantiate envs
    env_np = JAXBuckBoostConverterEnv(
        dt=args.dt,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_steps,
        grace_period_steps=args.grace_steps,
        target_voltage=args.target,
        # enforce_dcm=args.enforce_dcm,
        # >>> add these <<<
        # quantize_pwm=True,
        # quantize_mode="round",   # or "floor" if your Sim PWM floors
    )

    env_sim = BBCSimulinkEnv(
        model_name=args.model_name,
        dt=args.dt,
        frame_skip=args.frame_skip,
        max_episode_time=args.max_episode_time,
        grace_period_steps=args.grace_steps,
        target_voltage=args.target,
        random_target=False,
        use_fast_restart=True,
    )

    # Generate identical duty sequence
    t, duties = duty_sequence(
        mode=args.mode,
        steps=args.steps,
        dt=args.dt,
        frame_skip=args.frame_skip,
        center=args.center,
        amp=args.amp,
        freq_hz=args.freq_hz,
        constant_duty=args.constant_duty,
        duty_a=args.duty_a,
        duty_b=args.duty_b,
        step_at=args.step_at,
    )

    # Run and collect
    rec = run_one(env_np, env_sim, duties,
              sim_extra_steps=args.sim_extra_steps,
              sim_drop_last=args.sim_drop_last)


    # Summarize
    stats = summarize(rec, target_voltage=args.target, ma_window=args.ma_window)
    if stats:
        print("=== Summary (Simulink minus NumPy) ===")
        for k, v in stats.items():
            print(f"{k:>22s} : {v}")

    # ⬇️ Print measured duty stats if present
    md = rec.get("duty_meas_sim", None)
    if md is not None:
        import numpy as _np
        m = _np.asarray(md)
        m = m[_np.isfinite(m)]
        if m.size:
            tail = stats.get("ss_tail_len", max(50, len(m)//10))
            print(f"measured_duty_sim (mean): {m.mean():.4f}")
            print(f"measured_duty_sim (tail): {m[-tail:].mean():.4f}")
            print(f"duty_cmd (mean): {rec['duty'].mean():.4f}")

    # Save CSV and plots
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / args.csv_name
    df = save_csv(rec, out_csv)
    print(f"\nSaved telemetry: {out_csv.resolve()}  ({len(df)} rows)")

    make_plots(rec, target_voltage=args.target, title_prefix=f"{args.mode.upper()} duty", out_dir=args.out_dir, stats=stats)
    print(f"Saved plots to: {args.out_dir.resolve()}")

    # Cleanup
    env_np.close()
    env_sim.close()

if __name__ == "__main__":
    main()
