#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import numpy as np
import jax
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- Your local modules (names taken from your repo) ---
from cp_simulink_env import SimulinkEnv
from cp_numpy_wrapper import CartPoleGymWrapper as NumpyEnvWrapper
from cp_jax_wrapper    import CartPoleGymWrapper as JaxEnvWrapper
from cp_numpy          import CartPoleState as NumpyState
from cp_jax            import CartPoleState as JaxState

# Ensure JAX uses float64 like NumPy
jax.config.update("jax_enable_x64", True)

# ---------------- Utilities ----------------
def _flatten_numerics(obj):
    import numpy as _np
    if isinstance(obj, (list, tuple, _np.ndarray)):
        for el in obj:
            yield from _flatten_numerics(el)
    else:
        try:
            arr = _np.asarray(obj)
            if arr.size > 0 and _np.isfinite(arr.reshape(-1)[0]):
                yield float(arr.reshape(-1)[0])
        except Exception:
            return

def _parse_sim_obs(obs, debug=False):
    """Accept nested Simulink obs; return 4-vector [x, x_dot, theta, theta_dot] (radians, m, m/s)."""
    vals = list(_flatten_numerics(obs))
    if debug:
        print(f"[DEBUG] Simulink raw->flat: {vals}")
    if len(vals) >= 4:
        return np.array(vals[:4], dtype=np.float64), False
    if len(vals) >= 2:
        th, thd = vals[:2]
        return np.array([0.0, 0.0, th, thd], dtype=np.float64), True
    raise RuntimeError(f"Could not parse Simulink obs; flattened={vals} (raw type={type(obs)})")

def _maybe_set_attr(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)
        return True
    return False

def _install_params(env, params):
    """Try to install plant params into a gym-style wrapper in a defensive way."""
    applied = {}
    for k, v in params.items():
        ok = _maybe_set_attr(env, k, v)
        applied[k] = ok
    return applied

# -------------- Main Comparison --------------
def run(
    seed=42,
    steps=500,
    amp=2.0,
    omega=0.5,
    outdir="cp_compare_out",
    # Plant & integration to mirror Simulink
    dt=None,           # if None, use simulink_env.dt
    m_c=None,          # cart mass (kg)
    m_p=None,          # pole mass (kg)
    l=None,            # pole half-length or length depending on your models (keep consistent!)
    g=None,            # gravity (m/s^2)
    mu_c=None,         # cart friction
    mu_p=None,         # pole joint friction
    # Action handling
    clip_force=10.0,   # hard action saturation (±N)
    debug=False,
):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    print("--- CartPole Comparison: Simulink vs NumPy vs JAX ---")
    print(f"seed={seed}, steps={steps}, amp={amp} N, omega={omega} rad/s, clip_force=±{clip_force} N")
    print(f"Output -> {outdir.resolve()}\n")

    # 1) Initialize all envs
    try:
        sim = SimulinkEnv(model_name="PendCart", seed=seed)
    except Exception as e:
        print(f"Failed to init SimulinkEnv: {e}")
        sys.exit(1)

    np_env  = NumpyEnvWrapper(seed=seed, partial_obs=False)
    jx_env  = JaxEnvWrapper  (seed=seed, partial_obs=False)

    # 2) Resolve dt and plant parameters from Simulink (if exposed) or CLI
    sim_dt = getattr(sim, "dt", None)
    if dt is None:
        if sim_dt is None:
            print("[WARN] SimulinkEnv.dt not found and --dt not provided. Defaulting to 0.02s.")
            dt = 0.02
        else:
            dt = float(sim_dt)
    else:
        dt = float(dt)

    # Collect candidate params (prefer CLI if given; otherwise try to read from Simulink if it exposes them)
    def _pick(name, default=None):
        val = locals().get(name)  # capture closure value from outer scope (m_c, m_p, etc.)
        if val is not None:
            return float(val)
        return float(getattr(sim, name, default)) if getattr(sim, name, None) is not None else default

    params = {
        "m_c": _pick("m_c", None),
        "m_p": _pick("m_p", None),
        "l":   _pick("l",   None),
        "g":   _pick("g",   9.81),
        "mu_c":_pick("mu_c",0.0),
        "mu_p":_pick("mu_p",0.0),
        "dt":  dt,
    }

    # Install params into NumPy/JAX wrappers (best-effort; wrappers may store internally on .model or directly)
    applied_np   = _install_params(np_env, params)
    applied_jax  = _install_params(jx_env, params)

    # Also try common container locations (e.g., env.model or env.dyn)
    for container_name in ("model", "dyn", "physics"):
        if hasattr(np_env, container_name):
            _install_params(getattr(np_env, container_name), params)
        if hasattr(jx_env, container_name):
            _install_params(getattr(jx_env, container_name), params)

    # Loud report on parameter sync
    print("[PARAM SYNC]")
    print(f"  dt: simulink={sim_dt} -> using dt={dt}")
    for k, v in params.items():
        if k == "dt": continue
        print(f"  {k}={v}")
    print()

    # 3) Reset & align initial states
    sim_obs_raw, _ = sim.reset()
    sim_obs0, padded = _parse_sim_obs(sim_obs_raw, debug=debug)
    if padded:
        print("NOTE: Simulink returned only [theta, theta_dot]; padding x=x_dot=0 for comparison.")
    x0, xdot0, th0, thd0 = map(float, sim_obs0)

    np_env.reset(seed=seed); jx_env.reset(seed=seed)
    np_env.state = NumpyState(x=x0, x_dot=xdot0, theta=th0, theta_dot=thd0, t=0.0, done=False)
    jx_env.state = JaxState  (x=x0, x_dot=xdot0, theta=th0, theta_dot=thd0, t=0.0, done=False)
    _maybe_set_attr(np_env, "step_count", 0); _maybe_set_attr(jx_env, "step_count", 0)

    # 4) Sim loop (same scalar Newton force for all)
    T = np.arange(steps+1, dtype=np.float64) * dt
    U = np.clip(amp * np.sin(omega * T), -clip_force, clip_force)  # log exactly what we *intend* to send
    if debug:
        print("[DEBUG] First 10 forces (N):", np.round(U[:10], 6))

    sim_hist = [sim_obs0]
    np_hist  = [np.array([x0, xdot0, th0, thd0], dtype=np.float64)]
    jx_hist  = [np.array([x0, xdot0, th0, thd0], dtype=np.float64)]

    print(f"Running for {steps} steps at dt={dt}s…")
    for i in range(steps):
        u = float(U[i+1])  # force at next time; avoids phase lag vs. logging T
        # IMPORTANT: send identical Newton force to all environments
        s_raw, *_ = sim.step(np.array([u], dtype=np.float32))   # Simulink expects Newtons in your env
        n_obs, *_ = np_env.step(np.array([u], dtype=np.float32))
        j_obs, *_ = jx_env.step(np.array([u], dtype=np.float32))

        s_obs, _ = _parse_sim_obs(s_raw)
        sim_hist.append(s_obs)
        np_hist.append(np.asarray(n_obs, dtype=np.float64)[:4])
        jx_hist.append(np.asarray(j_obs, dtype=np.float64)[:4])

        if debug and (i < 6 or (i+1) % 100 == 0):
            print(f"[t={T[i+1]:.3f}] u={u:+.3f} -> "
                  f"Sim th={s_obs[2]:+.4f} rad, Num th={np_hist[-1][2]:+.4f}, JAX th={jx_hist[-1][2]:+.4f}")

    print("Simulation complete.\n")

    # 5) Close envs
    for e in (sim, np_env, jx_env):
        try: e.close()
        except Exception: pass

    # 6) Build outputs
    sim_hist = np.vstack(sim_hist)
    np_hist  = np.vstack(np_hist)
    jx_hist  = np.vstack(jx_hist)

    rad2deg = 180.0 / np.pi
    sim_deg = sim_hist.copy(); sim_deg[:, 2:] *= rad2deg
    np_deg  = np_hist.copy();  np_deg[:,  2:] *= rad2deg
    jx_deg  = jx_hist.copy();  jx_deg[:,  2:] *= rad2deg

    df = pd.DataFrame({
        "Time (s)": T,
        "Force (N)": U,
        "Sim x (m)": sim_hist[:, 0],  "NumPy x (m)": np_hist[:, 0],  "JAX x (m)": jx_hist[:, 0],
        "Sim ẋ (m/s)": sim_hist[:, 1], "NumPy ẋ (m/s)": np_hist[:, 1], "JAX ẋ (m/s)": jx_hist[:, 1],
        "Sim θ (°)": sim_deg[:, 2],   "NumPy θ (°)": np_deg[:, 2],    "JAX θ (°)": jx_deg[:, 2],
        "Sim θ̇ (°/s)": sim_deg[:, 3], "NumPy θ̇ (°/s)": np_deg[:, 3],  "JAX θ̇ (°/s)": jx_deg[:, 3],
    })
    csv_path = Path(outdir) / "cartpole_compare.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved table -> {csv_path}")

    # 7) Plots: force + states + angle diffs
    t = T
    fig, axs = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("CartPole: Simulink vs NumPy vs JAX (matched params, same force)", fontsize=16)

    axs[0].plot(t, U, label="Force (N)")
    axs[0].set_ylabel("u (N)"); axs[0].grid(True); axs[0].legend()

    axs[1].plot(t, sim_hist[:, 0], label="Simulink")
    axs[1].plot(t, np_hist[:, 0],  label="NumPy")
    axs[1].plot(t, jx_hist[:, 0],  label="JAX")
    axs[1].set_ylabel("x (m)"); axs[1].grid(True); axs[1].legend()

    axs[2].plot(t, sim_hist[:, 1], label="Simulink")
    axs[2].plot(t, np_hist[:, 1],  label="NumPy")
    axs[2].plot(t, jx_hist[:, 1],  label="JAX")
    axs[2].set_ylabel("ẋ (m/s)"); axs[2].grid(True); axs[2].legend()

    axs[3].plot(t, sim_deg[:, 2], label="Simulink")
    axs[3].plot(t, np_deg[:, 2],  label="NumPy")
    axs[3].plot(t, jx_deg[:, 2],  label="JAX")
    axs[3].set_ylabel("θ (°)"); axs[3].grid(True); axs[3].legend()

    d_np  = np.abs(sim_hist[:, 2] - np_hist[:, 2])
    d_jx  = np.abs(sim_hist[:, 2] - jx_hist[:, 2])
    axs[4].plot(t, d_np, label="|Sim - NumPy| (θ)")
    axs[4].plot(t, d_jx, label="|Sim - JAX| (θ)")
    axs[4].set_yscale("log"); axs[4].grid(True); axs[4].legend()
    axs[4].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0,0.03,1,0.95])
    fig_path = Path(outdir) / "cartpole_compare.png"
    plt.savefig(fig_path, dpi=160)
    print(f"Saved plot -> {fig_path}")

def main():
    p = argparse.ArgumentParser(description="Compare CartPole trajectories across Simulink/NumPy/JAX with matched params.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--amp", type=float, default=2.0, help="Sine force amplitude (N)")
    p.add_argument("--omega", type=float, default=0.5, help="Sine angular frequency (rad/s)")
    p.add_argument("--outdir", type=str, default="cp_compare_out")
    p.add_argument("--dt", type=float, default=None, help="Override dt (s); default uses SimulinkEnv.dt if available")
    p.add_argument("--m_c", type=float, default=None, help="Cart mass (kg)")
    p.add_argument("--m_p", type=float, default=None, help="Pole mass (kg)")
    p.add_argument("--l",   type=float, default=None, help="Pole length parameter (match convention across models)")
    p.add_argument("--g",   type=float, default=None, help="Gravity (m/s^2)")
    p.add_argument("--mu_c",type=float, default=None, help="Cart friction")
    p.add_argument("--mu_p",type=float, default=None, help="Pole joint friction")
    p.add_argument("--clip_force", type=float, default=10.0, help="Hard force clip ±N")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    run(
        seed=args.seed,
        steps=args.steps,
        amp=args.amp,
        omega=args.omega,
        outdir=args.outdir,
        dt=args.dt,
        m_c=args.m_c, m_p=args.m_p, l=args.l, g=args.g, mu_c=args.mu_c, mu_p=args.mu_p,
        clip_force=args.clip_force,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
