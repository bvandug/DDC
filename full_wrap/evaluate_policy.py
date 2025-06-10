import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from simulink_env import SimulinkEnv

def evaluate_custom_metrics(model, env, n_episodes=5, target_value=0.0, tolerance=0.01, stable_duration=0.5, sim_timestep=0.01):
    stabilisation_times = []
    steady_state_errors = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        t = 0.0
        ep_states = []
        stable_start_time = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            state_value = obs[0]  # Customize if your observation structure differs
            ep_states.append(state_value)

            # Stabilization detection
            if abs(state_value - target_value) < tolerance:
                if stable_start_time is None:
                    stable_start_time = t
                elif t - stable_start_time >= stable_duration:
                    break  # Considered stabilized
            else:
                stable_start_time = None

            t += sim_timestep

        # Stabilisation Time
        stab_time = stable_start_time if stable_start_time is not None else t
        stabilisation_times.append(stab_time)

        # Steady-State Error
        steady_state_errors.append(abs(ep_states[-1] - target_value))

    return stabilisation_times, steady_state_errors

# === Load Model & Env ===
env = SimulinkEnv()
model = TD3.load("td3_simulinker")

# === Basic Reward Evaluation ===
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# === Extended Metric Evaluation ===
stab_times, ss_errors = evaluate_custom_metrics(
    model,
    env,
    n_episodes=5,
    target_value=0.0,        # Your desired setpoint
    tolerance=0.01,          # Acceptable error band
    stable_duration=0.5,     # Seconds system must remain stable
    sim_timestep=0.01        # Match Simulink integration step
)

print(f"Mean Stabilisation Time: {np.mean(stab_times):.2f} s")
print(f"Mean Steady-State Error: {np.mean(ss_errors):.4f}")
