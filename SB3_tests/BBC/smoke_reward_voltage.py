#!/usr/bin/env python3
"""
Smoke test for sign correctness on voltage tracking.

- Probes whether the reward is sign-blind (|v| vs |vref|).
- Compares r(+30V) vs r(-30V) for target_voltage = -30V.
- Runs a short constant-duty rollout to see which sign the plant settles to.
"""

import numpy as np
from np_bbc_env import JAXBuckBoostConverterEnv

EPS = 1e-9

def reward_call(env, vc, prev_vc, duty, prev_duty):
    """Directly set internal state, then call the env's _reward(duty)."""
    env.vC = float(vc)
    env.prev_vC = float(prev_vc)
    env.prev_applied_duty = float(prev_duty)
    return env._reward(float(duty))

def compute_errors(vc, vref):
    """Both the current magnitude-based error (what your reward uses) and sign-aware error."""
    e_norm_mag   = abs(abs(vc) - abs(vref)) / max(abs(vref), EPS)           # current behaviour
    e_norm_sign  = abs(vc - vref) / max(abs(vref), EPS)                      # desired sign-aware
    return e_norm_mag, e_norm_sign

def symmetry_probe(env, vref=-30.0, duty=0.5):
    env.target_voltage = float(vref)

    # Keep progress and action-change at zero so we isolate r_track:
    prev_vc = +30.0
    r_pos = reward_call(env, vc=+30.0, prev_vc=prev_vc, duty=duty, prev_duty=duty)

    prev_vc = -30.0
    r_neg = reward_call(env, vc=-30.0, prev_vc=prev_vc, duty=duty, prev_duty=duty)

    e_mag_pos, e_sign_pos = compute_errors(+30.0, vref)
    e_mag_neg, e_sign_neg = compute_errors(-30.0, vref)

    print("=== Symmetry Probe (target -30 V) ===")
    print(f"r(+30V) = {r_pos:.6f} | e_norm_mag={e_mag_pos:.6f}, e_norm_sign={e_sign_pos:.6f}")
    print(f"r(-30V) = {r_neg:.6f} | e_norm_mag={e_mag_neg:.6f}, e_norm_sign={e_sign_neg:.6f}")
    if abs(r_pos - r_neg) < 1e-9:
        print(">> RESULT: Reward is SIGN-BLIND (|v| vs |vref|). +30V and -30V score the same.")
    else:
        print(">> RESULT: Reward distinguishes sign (good).")

def short_rollout(env, vref=-30.0, duty=0.5, steps=1000):
    obs, _ = env.reset()
    env.target_voltage = float(vref)

    v_hist = []
    for _ in range(steps):
        step_out = env.step(np.array([duty], dtype=np.float32))
        if len(step_out) == 5:  # Gymnasium
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:                   # Old Gym
            obs, reward, done, info = step_out

        v_hist.append(float(env.vC))
        if done:
            break


    v_hist = np.asarray(v_hist, dtype=float)
    v_last = v_hist[-100:].mean() if v_hist.size >= 100 else v_hist.mean()
    print("\n=== Short Rollout Check ===")
    print(f"Mean of last window: {v_last:.3f} V  (target {vref:.1f} V)")
    if np.sign(v_last) != np.sign(vref):
        print(">> WARNING: Plant converged to the WRONG SIGN relative to target.")
    else:
        print(">> OK: Plant output sign matches the target sign.")


def main():
    # Instantiate with defaults; this env is an INVERTING buck-boost (negative output).
    env = JAXBuckBoostConverterEnv(target_voltage=-30.0)

    # 1) Show reward symmetry (this will flag the abs(...) usage)
    symmetry_probe(env, vref=-30.0, duty=0.5)

    # 2) Do a quick constant-duty rollout (tweak duty if needed)
    short_rollout(env, vref=-30.0, duty=0.5, steps=1000)

if __name__ == "__main__":
    main()
