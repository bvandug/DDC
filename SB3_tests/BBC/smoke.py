# smoke_bbc_dynamics.py
#
# Quick, training-free checks for the inverting buck-boost dynamics.
# It runs fixed-duty episodes and compares the simulated output voltage
# against the ideal continuous-conduction approximation:
#     Vout ≈ -(D / (1 - D)) * Vin
#
# Usage (PowerShell):
#   python smoke_bbc_dynamics.py --frame-skip 20 --steps 2000 --plot --save-dir ./smoke_out
#
# Notes:
# - Uses RAW environment (no VecNormalize).
# - Assumes JAXBuckBoostConverterEnv is in np_bbc_env.py

import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from np_bbc_env import JAXBuckBoostConverterEnv


def ideal_vout_inverting_buckboost(vin: float, duty: float) -> float:
    """Ideal CCM mapping for inverting buck-boost: Vout = -(D/(1-D)) * Vin."""
    if duty <= 0.0 or duty >= 1.0:
        return float("nan")
    return - (duty / (1.0 - duty)) * vin


def ideal_hold_duty(vin: float, vref: float) -> float:
    """Duty that holds the reference magnitude in CCM: D* = |Vref|/(|Vref|+Vin)."""
    return abs(vref) / (abs(vref) + vin)


def run_fixed_duty_episode(env: JAXBuckBoostConverterEnv, duty: float, max_steps: int):
    """Run with a constant duty; also log effective duty and on_steps from env info when available."""
    obs, info = env.reset()
    rewards, duties, vC_list, iL_list = [], [], [], []
    eff_duty_list, on_steps_list, on_frac_list = [], [], []
    terminated_flag, truncated_flag = False, False

    fs_default = int(getattr(env, "frame_skip", 1))

    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step([duty])

        # Prefer telemetry from env (exact-duty mode). Fallback: quantized by rounding.
        if info is not None and ("eff_duty" in info or "on_steps" in info):
            fs_i = int(info.get("frame_skip", fs_default))
            if "eff_duty" in info:
                eff = float(info["eff_duty"])
            else:
                eff = float(info["on_steps"]) / float(fs_i)
            on  = int(info.get("on_steps", round(duty * fs_i)))
            ofr = float(info.get("on_frac", 0.0))
        else:
            on  = int(round(duty * fs_default))
            eff = on / float(fs_default)
            ofr = 0.0

        rewards.append(float(reward))
        duties.append(float(duty))
        vC_list.append(float(env.state[1]))
        iL_list.append(float(env.state[0]))
        eff_duty_list.append(eff)
        on_steps_list.append(on)
        on_frac_list.append(ofr)

        if terminated or truncated:
            terminated_flag, truncated_flag = bool(terminated), bool(truncated)
            break

    return {
        "rewards":   np.array(rewards, dtype=float),
        "duties":    np.array(duties, dtype=float),
        "eff_duty":  np.array(eff_duty_list, dtype=float),
        "on_steps":  np.array(on_steps_list, dtype=int),
        "on_frac":   np.array(on_frac_list, dtype=float),
        "vC":        np.array(vC_list, dtype=float),
        "iL":        np.array(iL_list, dtype=float),
        "terminated": terminated_flag,
        "truncated":  truncated_flag,
        "steps":       len(vC_list),
        "frame_skip":  fs_default,
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-skip", type=int, default=20, help="Control steps per PWM period (use 20 so one RL step = one 10 kHz PWM period).")
    parser.add_argument("--steps", type=int, default=2000, help="Maximum steps per scenario.")
    parser.add_argument("--target-voltage", type=float, default=-30.0, help="Negative target used by the environment.")
    parser.add_argument("--plot", action="store_true", help="Show plots.")
    parser.add_argument("--save-dir", type=str, default=None, help="Folder to save CSV and PNG files (optional).")
    args = parser.parse_args()

    # Build a raw environment (no normalization, no wrappers)
    env = JAXBuckBoostConverterEnv(
        max_episode_steps=args.steps,
        frame_skip=args.frame_skip,
        grace_period_steps=50,
        dt=5e-6,
        target_voltage=args.target_voltage,
    )

    vin = float(getattr(env, "Vin", 48.0))  # fall back to 48 V if not present
    vref = float(env.target_voltage)
    d_star = ideal_hold_duty(vin, vref)

    scenarios = [
        ("hold_duty_near_target", max(min(d_star, 0.9), 0.1)),
        ("low_duty_0p10", 0.10),
        ("moderate_duty_0p40", 0.40),  # near −30 V for Vin=48 V (ideal ~ −32 V)
        ("high_duty_0p60", 0.60),      # ideal ≈ −72 V; likely to hit safety bounds
    ]

    print(f"\nInput voltage (from env): {vin:.3f} V")
    print(f"Target voltage: {vref:.3f} V (negative expected)")
    print(f"Ideal hold duty for target: D* = {d_star:.4f}\n")

    all_results = {}

    for name, duty in scenarios:
        print(f"=== Scenario: {name} | fixed duty = {duty:.4f} ===")
        pred_vout = ideal_vout_inverting_buckboost(vin, duty)
        print(f"Ideal CCM prediction: Vout ≈ {pred_vout:.3f} V")

        res = run_fixed_duty_episode(env, duty, args.steps)
        all_results[name] = res

        final_vc = res["vC"][-1] if res["steps"] > 0 else float("nan")
        sign = "NEGATIVE" if final_vc < 0.0 else "POSITIVE or zero"
        print(f"Simulated final vC: {final_vc:.3f} V  (sign: {sign})")
        if res["terminated"]:
            print("Episode ended by TERMINATION (safety bound).")
        if res["truncated"]:
            print("Episode ended by TRUNCATION (time limit).")

        if res["steps"] > 0:
            eff = res["eff_duty"][-1]
            on  = res["on_steps"][-1]
            fs  = res["frame_skip"]
            print(f"Effective duty applied (quantized): {eff:.4f}  ({on}/{fs} substeps)\n")
        else:
            print("")

    
    # Optional save
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for name, res in all_results.items():
            csv_path = os.path.join(args.save_dir, f"{name}.csv")
            with open(csv_path, "w") as f:
                f.write("step,reward,duty_cmd,eff_duty,on_steps,on_frac,vC,iL\n")
                for i in range(res["steps"]):
                    f.write(
                        f"{i},{res['rewards'][i]},{res['duties'][i]},"
                        f"{res['eff_duty'][i]},{res['on_steps'][i]},{res['on_frac'][i]},"
                        f"{res['vC'][i]},{res['iL'][i]}\n"
                    )
            print(f"Saved {name} trace to: {csv_path}")

    # Optional plotting
    if args.plot:
        for name, res in all_results.items():
            steps = np.arange(res["steps"])
            plt.figure(figsize=(9, 6))
            plt.suptitle(f"{name}")

            # vC
            plt.subplot(3, 1, 1)
            plt.plot(steps, res["vC"])
            plt.axhline(y=vref, linestyle="--")
            plt.axhline(y=getattr(env, "V_OUT_MAX", abs(vref) * 1.5), linestyle=":")
            plt.axhline(y=-getattr(env, "V_OUT_MAX", abs(vref) * 1.5), linestyle=":")
            plt.axhline(y=getattr(env, "V_OUT_MIN", abs(vref) * 0.1), linestyle=":")
            plt.axhline(y=-getattr(env, "V_OUT_MIN", abs(vref) * 0.1), linestyle=":")
            plt.ylabel("vC [V]")

            # iL
            plt.subplot(3, 1, 2)
            plt.plot(steps, res["iL"])
            plt.ylabel("iL [A]")

            # duty
            plt.subplot(3, 1, 3)
            plt.plot(steps, res.get("eff_duty", res["duties"]))
            plt.ylabel("Eff. duty")
            plt.xlabel("Step")
            plt.ylabel("Duty")

            if args.save_dir:
                png = os.path.join(args.save_dir, f"{name}.png")
                plt.tight_layout()
                plt.savefig(png)
            else:
                plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()
