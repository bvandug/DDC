#!/usr/bin/env python3
import argparse, math
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Math predictors ----------------

def ideal_ccm_vout(Vin, D):
    if D <= 0.0 or D >= 1.0: return float("nan")
    return -Vin * D / (1.0 - D)

def predict_ccm_vout(Vin, R, L, D, Ron_sw, Ron_d, R_L, Vf):
    if D <= 0.0 or D >= 1.0: return float("nan")
    denom = 1.0 + (Ron_d + R_L) / (R * (1.0 - D)) + (D * Ron_sw) / (R * (1.0 - D)**2)
    absVo = ((D / (1.0 - D)) * Vin - Vf) / denom
    return -absVo

def predict_dcm_vout_triangle(Vin, R, L, Ts, D, Vf):
    if D <= 0.0 or D >= 1.0: return float("nan")
    term = Vf*Vf + 2.0 * (Vin*Vin) * (D*D) * Ts * (R / L)
    absVo = max(0.0, (-Vf + math.sqrt(term)) * 0.5)
    return -absVo

def dcm_resistive_Ipk(Vin, L, D, Ts, Ron_sw, R_L):
    R_on = Ron_sw + R_L
    if R_on <= 0:
        return Vin * D * Ts / L
    tau_on = L / R_on
    i_inf = Vin / R_on
    return i_inf * (1.0 - math.exp(-D * Ts / tau_on))

def predict_dcm_vout_resistive(Vin, R, L, Ts, D, Vf, Ron_sw, R_L, Ron_d, tol=1e-9, Vhi=500.0):
    Roff = Ron_d + R_L
    if Roff <= 0:
        return predict_dcm_vout_triangle(Vin, R, L, Ts, D, Vf)
    Ipk = dcm_resistive_Ipk(Vin, L, D, Ts, Ron_sw, R_L)
    def f(absVo):
        left = absVo / R
        i_inf_off = -(absVo + Vf) / Roff
        if Ipk <= 0.0:
            right = 0.0
        else:
            tau_off = L / Roff
            denom = -i_inf_off
            if denom <= 0: return left
            x = (Ipk - i_inf_off) / denom
            if x <= 1e-15: return left
            right = (tau_off / Ts) * (Ipk + i_inf_off * math.log(x))
        return left - right
    lo, hi = 0.0, float(Vhi)
    flo, fhi = f(lo), f(hi)
    if not (flo <= 0.0 and fhi >= 0.0):
        for s in [1000.0, 2000.0, 5000.0]:
            hi = s; fhi = f(hi)
            if flo <= 0.0 and fhi >= 0.0: break
        else:
            return predict_dcm_vout_triangle(Vin, R, L, Ts, D, Vf)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol: return -mid
        if fm > 0: hi = mid
        else:      lo = mid
    return -0.5 * (lo + hi)

def dcm_resistive_D2(absVo, Vin, L, Ts, D, Vf, Ron_sw, R_L, Ron_d):
    Roff = Ron_d + R_L
    if Roff <= 0: return float("nan")
    Ipk = dcm_resistive_Ipk(Vin, L, D, Ts, Ron_sw, R_L)
    tau_off = L / Roff
    denom = (absVo + Vf) / Roff
    if denom <= 0: return float("nan")
    t_z = tau_off * math.log(1.0 + Ipk / denom)
    return max(0.0, min(1.0, t_z / Ts))

# ---------------- Env glue ----------------

def import_env():
    from np_bbc_env import JAXBuckBoostConverterEnv as Env
    return Env, "jaxbb (np_bbc_env)"

def make_env(Env, dt, frame_skip, target):
    # EDIT HERE to toggle losses quickly:
    env = Env(
        dt=5e-6, frame_skip=26, target_voltage=target,
        Vin=48.0, L=220e-6, C=100e-6, R_load=5.1,
        # choose one of these blocks:
        # --- Lossless ---
        # Ron_sw=0.0, Ron_d=0.0, Vf=0.0, R_L=0.0, R_C=0.0
        # --- L/C zero only (keep switch/diode) ---
        Ron_sw=0.1, Ron_d=0.001, Vf=0.7, R_L=0.0, R_C=0.0
        # --- Realistic-ish example ---
        # Ron_sw=0.05, Ron_d=0.01, Vf=0.6, R_L=0.05, R_C=0.02
    )
    # safety mirror
    for attr, val in [
        ("R", 5.1), ("R_load", 5.1),
        ("Vin", 48.0), ("L", 220e-6), ("C", 100e-6),
    ]:
        if hasattr(env, attr): setattr(env, attr, val)
    env.dt = 5e-6; env.N = 26; env.frame_skip = 26; env.Tsw = env.dt * env.N
    if hasattr(env, "grace_period_steps"): env.grace_period_steps = 10_000_000
    return env

def quantize_duty(env, duty_cmd):
    N = int(getattr(env, "N", 26))
    dmin = float(getattr(env, "duty_min", 0.1)); dmax = float(getattr(env, "duty_max", 0.9))
    duty_cmd = float(np.clip(duty_cmd, dmin, dmax))
    rounding = getattr(env, "pwm_rounding", "round").lower()
    if rounding == "floor": k_on = int(np.floor(duty_cmd * N + 1e-12))
    else:                    k_on = int(np.round(duty_cmd * N))
    k_on = min(max(k_on, int(np.ceil(dmin*N))), int(np.floor(dmax*N)))
    return k_on, k_on / N

# ---------------- One-period probe & mode classification ----------------

def probe_one_period(env, duty_cmd):
    k_on, duty_eff = quantize_duty(env, duty_cmd)
    N = env.N
    iL = float(getattr(env, "iL", 0.0))
    uC = float(getattr(env, "uC", 0.0))
    vC = float(getattr(env, "vC", 0.0))
    i_hist, v_hist = [], []
    zero_cross_k = None
    for k in range(N):
        if k < k_on:
            iL, uC, vC = env._on_step(iL, uC, env.dt)
        else:
            prev = iL
            iL, uC, vC = env._off_step(iL, uC, env.dt)
            if zero_cross_k is None and prev > 0.0 and iL <= 0.0:
                zero_cross_k = k
        i_hist.append(iL); v_hist.append(vC)
    i_hist = np.array(i_hist, dtype=float)
    v_hist = np.array(v_hist, dtype=float)
    Ipk = float(np.max(i_hist)) if i_hist.size else float("nan")
    Imin = float(np.min(i_hist)) if i_hist.size else float("nan")
    if zero_cross_k is None:
        d2_steps = max(0, N - k_on)
    else:
        d2_steps = max(0, zero_cross_k - k_on + 1)
    D2 = d2_steps / float(N)
    return {"applied_duty": duty_eff, "k_on": k_on, "N": N,
            "I_pk": Ipk, "I_min": Imin, "D2": float(D2),
            "v_end": float(v_hist[-1]) if v_hist.size else float("nan")}

def classify_mode(env, duty_cmd):
    k_on, _ = quantize_duty(env, duty_cmd)
    N = env.N
    iL = float(getattr(env, "iL", 0.0))
    uC = float(getattr(env, "uC", 0.0))
    vC = float(getattr(env, "vC", 0.0))
    Imin = float("inf")
    for k in range(N):
        if k < k_on:
            iL, uC, vC = env._on_step(iL, uC, env.dt)
        else:
            iL, uC, vC = env._off_step(iL, uC, env.dt)
        Imin = min(Imin, iL)
    return ("CCM" if Imin > 0.0 else "DCM"), Imin

# ---------------- Long-run constant duty ----------------

def run_const_duty(env, duty_cmd, steps, ma_tail=25):
    obs, _ = env.reset()
    v_end_hist = []
    v_mean_hist = []
    t_hist = []
    term = trunc = False
    info_last = {}
    Ts = env.dt * env.N
    for n in range(steps):
        obs, r, term, trunc, info = env.step(np.array([duty_cmd], dtype=np.float32))
        v_end = info.get("vC", float("nan"))
        v_avg = info.get("vC_mean_period", v_end)  # fallback if env not patched
        v_end_hist.append(v_end)
        v_mean_hist.append(v_avg)
        t_hist.append((n+1) * Ts)
        info_last = info
        if term or trunc: break
    v_end_hist = np.array(v_end_hist, dtype=float)
    v_mean_hist = np.array(v_mean_hist, dtype=float)
    t_hist = np.array(t_hist, dtype=float)

    # old-style "steady state" as moving average of last K steps (end-of-period)
    v_ma = v_end_hist[-ma_tail:].mean() if len(v_end_hist) >= ma_tail else (
           v_end_hist.mean() if len(v_end_hist)>0 else float("nan"))
    return {
        "v_ss": float(v_ma),
        "applied_duty": info_last.get("eff_duty", float("nan")),
        "steps": len(v_end_hist), "terminated": term, "truncated": trunc,
        "last_v": float(v_end_hist[-1]) if len(v_end_hist) else float("nan"),
        "t": t_hist, "v_end": v_end_hist, "v_avg": v_mean_hist
    }

# ---------------- Plotting ----------------

def plot_series(ax, t, v_end, v_avg, label, preds=None):
    t_ms = t * 1e3
    ax.plot(t_ms, v_end, lw=1.0, label=f"{label} (end)")
    ax.plot(t_ms, v_avg, lw=1.5, linestyle="--", label=f"{label} (period-mean)")
    if preds:
        for (name, val) in preds:
            if val == val:  # not NaN
                ax.hlines(val, t_ms[0] if t_ms.size else 0, t_ms[-1] if t_ms.size else 1,
                          linestyles=":", linewidth=1.0, label=name)
    ax.set_xlabel("time [ms]"); ax.set_ylabel("Vout [V]")
    ax.grid(True)

# ---------------- CLI ----------------

def main():
    p = argparse.ArgumentParser(description="Constant-duty sweep with predictors, internals, and optional plots.")
    p.add_argument("--dt", type=float, default=5e-6)
    p.add_argument("--frame-skip", type=int, default=26)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--target", type=float, default=-30.0)
    p.add_argument("--duty", type=float, default=None)
    p.add_argument("--sweep", nargs="*", type=float, default=None)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--plot", action="store_true", help="Show matplotlib plots")
    p.add_argument("--save", type=str, default=None, help="Save plot to this PNG path")
    args = p.parse_args()

    Env, name = import_env()
    env = make_env(Env, args.dt, args.frame_skip, args.target)

    Vin = getattr(env, "Vin", 48.0); L = getattr(env, "L", 220e-6); C = getattr(env, "C", 100e-6)
    R = getattr(env, "R", getattr(env, "R_load", 5.1)); Ts = env.dt * env.N
    Ron_sw = getattr(env, "Ron_sw", 0.1); Ron_d = getattr(env, "Ron_d", 0.001)
    R_L = getattr(env, "R_L", 1.0); Vf = getattr(env, "Vf", 0.7)

    def run_case(label, D_cmd, ax=None):
        # Long-run sim
        res = run_const_duty(env, D_cmd, args.steps, ma_tail=25)
        D_eff = float(res["applied_duty"]); V_sim = float(res["v_ss"])

        # Mode detection and predictors
        mode, Imin = classify_mode(env, D_cmd)
        V_ideal = ideal_ccm_vout(Vin, D_eff)
        V_ccm   = predict_ccm_vout(Vin, R, L, D_eff, Ron_sw, Ron_d, R_L, Vf)
        V_dcm_t = predict_dcm_vout_triangle(Vin, R, L, Ts, D_eff, Vf)
        V_dcm_r = predict_dcm_vout_resistive(Vin, R, L, Ts, D_eff, Vf, Ron_sw, R_L, Ron_d)

        print(f"\n=== {label} ===")
        k_on = int(round(D_eff*env.N))
        print(f"Command D={D_cmd:.6f}  | Applied D={D_eff:.6f}  (k_on={k_on}/{env.N})")
        print(f"Params: Vin={Vin:.3f} V, L={L*1e6:.1f} µH, C={C*1e6:.1f} µF, R={R:.3f} Ω, Ts={Ts*1e6:.1f} µs")
        print(f"Ideal CCM (lossless)      : Vout ≈ {V_ideal: .3f} V")
        print(f"CCM (lossy, predicted)    : Vout ≈ {V_ccm: .3f} V")
        if mode == "DCM":
            print(f"DCM (triangle, predicted) : Vout ≈ {V_dcm_t: .3f} V")
            print(f"DCM (resistive, predicted): Vout ≈ {V_dcm_r: .3f} V")
        print(f"Sim: v_ss={V_sim: .3f} V  term={res['terminated']} trunc={res['truncated']}  steps={res['steps']}")

        if args.probe:
            intern = probe_one_period(env, D_cmd)
            print("---- One-period internals ----")
            print(f"Applied duty: {intern['applied_duty']:.6f}  (k_on={intern['k_on']}/{intern['N']})")
            print(f"I_pk={intern['I_pk']:.6f} A   I_min={intern['I_min']:.6f} A   D2_meas≈{intern['D2']:.6f}")
            if mode == "DCM":
                absVo = abs(V_dcm_r) if V_dcm_r == V_dcm_r else abs(V_dcm_t)
                D2p = dcm_resistive_D2(absVo, Vin, L, Ts, D_eff, Vf, Ron_sw, R_L, Ron_d)
                if D2p == D2p:
                    print(f"D2_pred (resistive)≈{D2p:.6f}")

        # plotting
        if args.plot and res["t"].size > 0 and ax is not None:
            preds = [("Ideal (CCM)", V_ideal), ("CCM (lossy)", V_ccm)]
            if mode == "DCM":
                preds += [("DCM (triangle)", V_dcm_t), ("DCM (resistive)", V_dcm_r)]
            plot_series(ax, res["t"], res["v_end"], res["v_avg"], f"D={D_eff:.4f}", preds=preds)
            ax.set_title(label)

    print(f"Vin={Vin:.3f} V  Target={args.target:.3f} V  (env='{name}')")

    if args.sweep:
        n = len(args.sweep)
        if args.plot:
            fig, axes = plt.subplots(n, 1, figsize=(8, 3.2*n), sharex=True)
            if n == 1: axes = [axes]
        else:
            axes = [None]*n
        for i, D in enumerate(args.sweep):
            run_case(f"Sweep D={D:.6f}", D, ax=axes[i])
        if args.plot:
            axes[-1].set_xlabel("time [ms]")
            axes[0].legend(loc="best")
            fig.tight_layout()
            if args.save:
                plt.savefig(args.save, dpi=160)
            plt.show()
    else:
        if args.plot:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        else:
            axes = [None, None]
        D_star = abs(args.target) / (abs(args.target) + Vin) if args.duty is None else float(args.duty)
        run_case("Case A: near target", D_star, ax=axes[0])
        run_case("Case B: low duty (0.10)", 0.10, ax=axes[1])
        if args.plot:
            axes[-1].set_xlabel("time [ms]")
            axes[0].legend(loc="best")
            plt.tight_layout()
            if args.save:
                plt.savefig(args.save, dpi=160)
            plt.show()

if __name__ == "__main__":
    main()
