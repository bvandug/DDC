import argparse
import importlib.util
import math
import os
import sys
import csv

import numpy as np

def approx_equal(a, b, tol):
    """ Return True if two scalars are within an absolute tolerance.

        Parameters
        ----------
        a : float
            First value.
        b : float
            Second value.
        tol : float
            Absolute tolerance on |a - b|.

        Returns
        -------
        bool
            True iff |a - b| ≤ tol.
    """

    return abs(a - b) <= tol

def load_env_class(path, class_name=None):
    """ Dynamically load a Gymnasium-like env class from a Python file.

        If `class_name` is provided, returns that attribute from the loaded
        module and raises if not found. Otherwise, auto-detects the first class
        object that has `step`, `reset`, and `observation_space` attributes.

        Parameters
        ----------
        path : str | os.PathLike
            Filesystem path to the `.py` module containing the env class.
        class_name : str | None, optional
            Explicit class name to load. If None, attempt auto-detection.

        Returns
        -------
        type
            The environment class object.

        Raises
        ------
        AttributeError
            If `class_name` is given but not found in the module.
        RuntimeError
            If no suitable env class can be auto-detected.
    """

    spec = importlib.util.spec_from_file_location("user_env_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if class_name:
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise AttributeError(f"Class '{class_name}' not found in {path}")
        return cls
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and hasattr(obj, "step") and hasattr(obj, "reset") and hasattr(obj, "observation_space"):
            return obj
    raise RuntimeError("Could not find an env class (with step/reset/observation_space) in the module.")

def simulate_constant_duty(env, duty, n_steps=2500, warmup=1500):
    """ Run the env at a fixed duty cycle and estimate steady-state outputs.

        Performs an initial rollout (collecting termination stats), then a
        warmup + windowed averaging phase to estimate steady-state capacitor
        voltage and inductor current.

        Parameters
        ----------
        env :
            An instantiated env supporting `reset()` and `step(a)` with duty `a`.
        duty : float
            Constant duty cycle command in [0, 1].
        n_steps : int, optional
            Steps to run for the initial pass (termination detection).
        warmup : int, optional
            Steps to discard before averaging in the second pass.

        Returns
        -------
        dict
            Keys:
            - `"vC_mean"` : float — mean capacitor voltage over the window.
            - `"iL_mean"` : float — mean inductor current over the window (NaN
            if not available).
            - `"terminations"` : int — count of terminations in the first pass.
            - `"truncations"` : int — count of truncations in the first pass.
    """

    obs, info = env.reset()
    if hasattr(env, "state"):
        env.state = np.array([0.0, 0.0], dtype=float)
    terminations = 0
    truncations = 0
    for _ in range(n_steps):
        a = np.array([duty], dtype=np.float32)
        obs, rew, term, trunc, info = env.step(a)
        if term or trunc:
            terminations += int(term)
            truncations += int(trunc)
            if term:
                break
            
    window_vC, window_iL = [], []
    obs, info = env.reset()
    if hasattr(env, "state"):
        env.state = np.array([0.0, 0.0], dtype=float)
    for k in range(warmup + 400):
        a = np.array([duty], dtype=np.float32)
        obs, rew, term, trunc, info = env.step(a)
        if k >= warmup:
            if hasattr(env, "state"):
                vC = float(env.state[1])
                iL = float(env.state[0])
            else:
                vC = float(obs[0]); iL = float("nan")
            window_vC.append(vC); window_iL.append(iL)
            if term or trunc:
                break
    vC_mean = float(np.mean(window_vC)) if window_vC else float("nan")
    iL_mean = float(np.mean(window_iL)) if window_iL else float("nan")
    return {"vC_mean": vC_mean, "iL_mean": iL_mean, "terminations": terminations, "truncations": truncations}

def Lcrit(R, D, fsw):
    """Compute the CCM boundary inductance for an inverting buck–boost.

        Defined as:
            L_crit = R * (1 - D)^2 / (2 * f_sw)

        Parameters
        ----------
        R : float
            Load resistance (ohms).
        D : float
            Duty cycle in [0, 1].
        fsw : float
            Switching frequency (Hz).

        Returns
        -------
        float
            Critical inductance (henries) at which operation is at the CCM/DCM boundary.
    """

    return R * (1.0 - D)**2 / (2.0 * fsw)

def main():
    """ CLI entry point: run smoke tests and CCM coverage on a buck–boost env.

        Parses command-line arguments, dynamically loads the env class, executes
        a suite of sanity checks (step-period, OFF-phase signs, DCM clamp,
        steady-state ratio vs. ideal, safety termination, derivative check, and
        optional reward monotonicity), prints a CCM coverage table, and reports
        PASS/FAIL counts.

        Side Effects
        -----------
        - Prints test results and summaries to stdout.
        - Optionally writes a CSV of duty sweep results if `--csv-out` is used.

        Returns
        -------
        None
    """

    p = argparse.ArgumentParser()
    p.add_argument("--env-file", required=True, help="Path to env .py file (e.g., np_bbc_env.py)")
    p.add_argument("--class", dest="class_name", default=None, help="Env class name (auto-detect if omitted)")
    p.add_argument("--duties", default="0.25,0.35,0.45,0.55", help="Comma-separated duty list")
    p.add_argument("--ratio-low", type=float, default=0.5, help="Min acceptable |vC|/ideal ratio")
    p.add_argument("--ratio-high", type=float, default=1.2, help="Max acceptable |vC|/ideal ratio")
    p.add_argument("--csv-out", default="", help="Optional CSV output path for duty sweep results")
    args = p.parse_args()

    EnvClass = load_env_class(args.env_file, args.class_name)
    env = EnvClass()

    passes = 0; fails = 0
    def report(name, passed, details=""):
        nonlocal passes, fails
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}" + (f"  — {details}" if details else ""))
        if passed: passes += 1
        else: fails += 1

    try:
        obs, info = env.reset()
        t0 = getattr(env, "time", None)
        a = np.array([0.3], dtype=np.float32)
        obs, rew, term, trunc, info = env.step(a)
        t1 = getattr(env, "time", None)
        dt = getattr(env, "dt", None)
        fs = getattr(env, "frame_skip", None)
        if t0 is not None and t1 is not None and dt is not None and fs is not None:
            period = (t1 - t0); expected = dt * fs
            report("Step advances one PWM period", approx_equal(period, expected, max(1e-12, expected*1e-6)),
                   f"Δt={period} s, expected {expected} s")
        else:
            report("Step advances one PWM period", False, "Missing attributes 'time', 'dt', or 'frame_skip'")
    except Exception as e:
        report("Step advances one PWM period", False, str(e))

    if hasattr(env, "_integrate_substep"):
        try:
            iL0, vC0 = 2.0, -20.0
            iL1, vC1 = env._integrate_substep(0, iL0, vC0)
            di = iL1 - iL0; dv = vC1 - vC0
            report("OFF-phase signs (di<0, dv<0)", (di < 0.0 and dv < 0.0), f"di={di:.6g}, dv={dv:.6g}")
        except Exception as e:
            report("OFF-phase signs (di<0, dv<0)", False, str(e))
    else:
        report("OFF-phase signs (di<0, dv<0)", False, "_integrate_substep not available")

    if hasattr(env, "_integrate_substep"):
        try:
            iL0, vC0 = 0.01, -10.0
            iL1, vC1 = env._integrate_substep(0, iL0, vC0)
            report("DCM clamp (iL >= 0)", iL1 >= -1e-9, f"iL_new={iL1:.6g}")
        except Exception as e:
            report("DCM clamp (iL >= 0)", False, str(e))
    else:
        report("DCM clamp (iL >= 0)", False, "_integrate_substep not available")

    # 4) Constant-duty sweep vs ideal + CCM coverage table
    duty_list = [float(x) for x in args.duties.split(",")]
    Vin = getattr(env, "Vin", 48.0)
    R   = getattr(env, "R_load", getattr(env, "R", 20.0))
    dt  = getattr(env, "dt", None)
    fs  = getattr(env, "frame_skip", None)
    fsw = (1.0 / (dt * fs)) if (dt and fs) else 10_000.0
    Lenv = getattr(env, "L", float("nan"))

    rows = []
    try:
        all_ok = True
        messages = []
        for D in duty_list:
            Lc = Lcrit(R, D, fsw)
            sim = simulate_constant_duty(env, D, n_steps=2200, warmup=1500)
            vC_mean = sim["vC_mean"]
            ideal = (D / (1.0 - D)) * Vin
            ratio = abs(vC_mean) / ideal if ideal != 0 else float("nan")
            sign_ok = (vC_mean < 0.0)

            rows.append({
                "D": D,
                "Lcrit_H": Lc,
                "Lcrit_uH": Lc * 1e6,
                "L_env_uH": Lenv * 1e6,
                "CCM_at_D": (Lenv >= Lc),
                "mean_vC": vC_mean,
                "ideal_mag": ideal,
                "ratio(|vC|/ideal)": ratio,
                "sign_negative": sign_ok,
                "terminated": bool(sim["terminations"]),
                "truncated": bool(sim["truncations"]),
            })

            if not sign_ok or not (args.ratio_low <= ratio <= args.ratio_high):
                all_ok = False
                messages.append(f"D={D}: sign_ok={sign_ok}, ratio={ratio:.3f}")

        # Print CCM coverage table
        print("\nCCM Coverage (Lcrit vs L_env):")
        print(" D     Lcrit[µH]   L_env[µH]   CCM_at_D")
        for r in rows:
            print(f" {r['D']:0.2f}   {r['Lcrit_uH']:9.2f}   {r['L_env_uH']:9.2f}   {str(r['CCM_at_D']):>8}")

        # Aggregate CCM across sweep
        ccm_all = all(r["CCM_at_D"] for r in rows)
        print(f"\nCCM across duty sweep: {ccm_all}")

        report("Const-duty steady-state close to ideal",
               all_ok, "; ".join(messages) if messages else "All within tolerance")
    except Exception as e:
        report("Const-duty steady-state close to ideal", False, str(e))

    try:
        obs, info = env.reset()
        if hasattr(env, "grace_period_steps"):
            env.current_step = env.grace_period_steps + 1
        if hasattr(env, "state"):
            Vmax = getattr(env, "V_OUT_MAX", 45.0)
            env.state = np.array([0.0, - (Vmax + 5.0)], dtype=float)
        obs, rew, term, trunc, info = env.step(np.array([0.5], dtype=np.float32))
        report("Safety termination (over-voltage magnitude)", bool(term), f"terminated={term}, reward={float(rew):.2f}")
    except Exception as e:
        report("Safety termination (over-voltage magnitude)", False, str(e))

    try:
        obs, info = env.reset()
        prev_error = float(obs[1])
        obs2, rew, term, trunc, info = env.step(np.array([0.5], dtype=np.float32))
        error = float(obs2[1]); d_err = float(obs2[2])
        dt_step = getattr(env, "dt", 1.0) * getattr(env, "frame_skip", 1)
        expected = (error - prev_error) / dt_step
        tol = max(1e-6, abs(expected)*1e-6)
        report("Obs derivative matches finite difference", abs(d_err - expected) <= tol,
               f"d_error={d_err:.6g}, expected={expected:.6g}")
    except Exception as e:
        report("Obs derivative matches finite difference", False, str(e))

    if hasattr(env, "calculate_reward"):
        try:
            vref = abs(getattr(env, "target_voltage", -30.0))
            env.prev_state = np.array([0.0, - (vref + 10.0)], dtype=float)
            env.prev_duty = 0.5
            candidates = [- (vref + 5.0), - (vref + 1.0), - (vref - 0.1)]
            rewards = []
            for vC in candidates:
                env.state = np.array([0.0, vC], dtype=float)
                r = env.calculate_reward(0.5)
                rewards.append(float(r))
            passed = (rewards[0] <= rewards[1] + 1e-6) and (rewards[1] <= rewards[2] + 1e-6)
            report("Reward increases as |vC|→|Vref|", passed,
                   f"r_far={rewards[0]:.3f}, r_near={rewards[1]:.3f}, r_inside={rewards[2]:.3f}")
        except Exception as e:
            report("Reward increases as |vC|→|Vref|", False, str(e))
    else:
        report("Reward increases as |vC|→|Vref|", False, "calculate_reward not found")

    print("\nSummary: {} PASS / {} FAIL".format(passes, fails))

if __name__ == "__main__":
    main()
