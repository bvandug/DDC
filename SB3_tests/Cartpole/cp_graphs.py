#!/usr/bin/env python3
# Runs 1 episode for DQN and TD3 from a fixed initial angle and logs time/angle to CSV.

import os, csv, glob
import numpy as np

from cp_simulink_env import SimulinkEnv, DiscretizedActionWrapper  # your classes
from stable_baselines3 import DQN, TD3

# ========= USER SETTINGS (edit these 3 lines if needed) =======================
INIT_ANGLE_RAD = -0.944298505783081                 # exact start angle you asked for
MODEL_ROOT     = os.path.join("CP_JAX", "jax_clean") # folder containing run subdirs
DT             = 0.01                               # 1 ms sampling like your example
MAX_TIME       = 5.0                                 # seconds (change if you want shorter)
# ==============================================================================

def find_best_model(root, token):
    """
    Find a 'best_model.zip' inside a subfolder whose name contains token (e.g. 'DQN'/'TD3').
    Returns the first match; raise if none is found.
    """
    token = token.upper()
    candidates = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Model root not found: {root}")
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub): 
            continue
        if token not in name.upper(): 
            continue
        zip_path = os.path.join(sub, "best_model.zip")
        if os.path.isfile(zip_path):
            candidates.append(zip_path)
    if not candidates:
        # also look deeper: */*/best_model.zip (optional fallback)
        for zip_path in glob.glob(os.path.join(root, "**", "best_model.zip"), recursive=True):
            if token in os.path.basename(os.path.dirname(zip_path)).upper() or token in os.path.basename(os.path.dirname(os.path.dirname(zip_path))).upper():
                candidates.append(zip_path)
    if not candidates:
        raise FileNotFoundError(f"No best_model.zip found for {token} under {root}")
    return candidates[0]

def reinit_with_angle(env: SimulinkEnv, theta0: float):
    """
    Force the Simulink model inside env to the desired initial angle and refresh xFinal.
    Matches the approach used in your env reset logic but with a fixed θ0.
    """
    # point the block's 'init' parameter to our exact starting angle
    env.eng.set_param(f"{env.model_name}/Pendulum and Cart", "init", str(theta0), nargout=0)

    # re-create xFinal at t≈0 so future steps start from (theta0, ·)
    env.eng.set_param(env.model_name, "FastRestart", "off", nargout=0)
    env.eng.eval(
        f"out = sim('{env.model_name}', 'LoadInitialState','off', "
        f"'SaveFinalState','on', 'StateSaveName','xFinal', 'StopTime','1e-4'); "
        f"xFinal = out.xFinal;",
        nargout=0
    )
    env.eng.set_param(env.model_name, "FastRestart", "on", nargout=0)
    # keep env.current_time at 0 so first control step lands at 0.001 s

def run_one_episode_to_csv(model, env: SimulinkEnv, csv_path: str, theta0: float):
    """
    Roll out exactly one episode at 1 ms, logging 'time_s,angle_rad' rows.
    The first row is '0.000, theta0' to match your example.
    """
    # Reset once to boot the engine, then overwrite the initial condition precisely
    env.reset(seed=42)
    reinit_with_angle(env, theta0)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "angle_rad"])
        # initial line exactly at t=0.0 using the angle we set
        w.writerow([0.0, float(theta0)])

        done = False
        while not done:
            # SB3 predict (deterministic)
            action, _ = model.predict(None, deterministic=True) if model.__class__ is DQN.__class__ else model.predict(None, deterministic=True)
            # The env ignores 'None'; we need the latest obs, so we query it from Simulink via step progression.
            # -> Better: keep track of last obs. We'll just step with the last observation we saw.
            # Tiny shim: on first call, we request the current observation by stepping with zero torque once, then revert.
            break

    # A cleaner, explicit loop with the current observation:
def run_episode_with_obs(model, env: SimulinkEnv, csv_path: str, theta0: float):
    env.reset(seed=42)
    reinit_with_angle(env, theta0)

    # we need an initial observation; do a zero-duration peek by stepping one dt with zero action,
    # but we already wrote the t=0 row, so we proceed directly (the first real row will be at 0.001 s).

    rows = []
    # loop until SimulinkEnv signals 'done' (angle out of bounds or time limit)
    done = False
    last_obs = np.array([theta0, 0.0], dtype=np.float32)  # angle, angular velocity (approx start)
    while not done:
        action, _ = model.predict(last_obs, deterministic=True)
        # step; cp_simulink_env returns legacy (obs, reward, done, info)
        obs, reward, done, info = env.step(action)
        t = float(info.get("time", env.current_time))
        theta = float(obs[0])
        rows.append((t, theta))
        last_obs = obs

    # write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "angle_rad"])
        w.writerow([0.0, float(theta0)])
        for t, theta in rows:
            # format to 3 decimals for millisecond grid; keeps 0.009 from becoming 0.009000000000000001
            w.writerow([round(t, 3), theta])

def main():
    # --- TD3 (continuous actions) ---
    td3_env = SimulinkEnv(dt=DT, max_episode_time=MAX_TIME, eval_obs_noise_std=0.0)
    td3_path = find_best_model(MODEL_ROOT, "TD3")
    td3 = TD3.load(td3_path)
    run_episode_with_obs(td3, td3_env, "td3_cartpole_trace.csv", INIT_ANGLE_RAD)
    td3_env.close()
    print("Wrote td3_cartpole_trace.csv")

    # --- DQN (discretized continuous actions) ---
    dqn_env_cont = SimulinkEnv(dt=DT, max_episode_time=MAX_TIME, eval_obs_noise_std=0.0)
    # map to 5 evenly spaced torques [-Fmax, ..., +Fmax] as in your eval script
    max_torque = float(dqn_env_cont.action_space.high[0])
    torque_values = np.linspace(-max_torque, max_torque, 5)
    dqn_env = DiscretizedActionWrapper(dqn_env_cont, force_values=torque_values)

    dqn_path = find_best_model(MODEL_ROOT, "DQN")
    dqn = DQN.load(dqn_path)
    run_episode_with_obs(dqn, dqn_env, "dqn_cartpole_trace.csv", INIT_ANGLE_RAD)
    dqn_env.close()
    print("Wrote dqn_cartpole_trace.csv")

if __name__ == "__main__":
    main()
