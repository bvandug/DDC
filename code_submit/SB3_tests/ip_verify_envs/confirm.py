import sys
import numpy as np
import jax
import matplotlib.pyplot as plt
import pandas as pd  # for nice table formatting

from ip_simulink_env import SimulinkEnv
from ip_numpy_wrapper import InvertedPendulumGymWrapper as NumpyEnvWrapper
from ip_jax_wrapper import InvertedPendulumGymWrapper as JaxEnvWrapper
from ip_numpy import PendulumState as NumpyPendulumState
from ip_jax import PendulumState as JaxPendulumState

# make sure JAX uses 64‑bit to match NumPy
jax.config.update("jax_enable_x64", True)

def run_comparison_test(seed=42, num_steps=500):
    print("--- Starting Comparison: Simulink vs. NumPy vs. JAX ---")
    print(f"Initializing all environments with seed: {seed}...")

    # 1. Initialize all three envs
    try:
        simulink_env = SimulinkEnv(model_name="pendulum", seed=seed)
    except Exception as e:
        print(f"Failed to init SimulinkEnv: {e}")
        sys.exit(1)

    numpy_env = NumpyEnvWrapper(seed=seed)
    jax_env   = JaxEnvWrapper(seed=seed)
    print("All environments initialized.\n")

    # 2. Reset and sync initial state
    print("Resetting environments and syncing initial state...")
    simulink_obs = simulink_env.reset()
    theta0, theta_dot0 = simulink_obs
    numpy_env.state = NumpyPendulumState(theta=theta0, theta_dot=theta_dot0, t=0.0, done=False)
    jax_env.state   = JaxPendulumState   (theta=theta0, theta_dot=theta_dot0, t=0.0, done=False)

    simulink_history = [simulink_obs]
    numpy_history    = [np.array([theta0, theta_dot0], dtype=np.float64)]
    jax_history      = [jax_env._obs()]

    # 3. Run the loop
    print(f"Running simulation for {num_steps} steps...")
    for i in range(num_steps):
        u = np.array([np.sin(i * 0.1) * 2.0], dtype=np.float32)
        s_obs, *_ = simulink_env.step(u)
        n_obs, *_ = numpy_env.step(u)
        j_obs, *_ = jax_env.step(u)

        simulink_history.append(s_obs)
        numpy_history   .append(n_obs)
        jax_history     .append(j_obs)

        if (i+1) % 100 == 0:
            print(f"  ... completed step {i+1}")
    print("Simulation finished.\n")

    # 4. Cleanup
    print("Closing all environments...")
    simulink_env.close()
    numpy_env.close()
    jax_env.close()
    print("Cleanup complete.\n")

    # 5. Convert to degrees & tabulate
    simulink_deg = np.array(simulink_history) * 180/np.pi
    numpy_deg    = np.array(numpy_history)    * 180/np.pi
    jax_deg      = np.array(jax_history)      * 180/np.pi

    df = pd.DataFrame({
        'Time (s)':                np.arange(num_steps+1) * simulink_env.dt,
        'Simulink θ (°)':          simulink_deg[:, 0],
        'NumPy θ (°)':             numpy_deg[:, 0],
        'JAX θ (°)':               jax_deg[:, 0],
        'Simulink θ̇ (°/s)':        simulink_deg[:, 1],
        'NumPy θ̇ (°/s)':           numpy_deg[:, 1],
        'JAX θ̇ (°/s)':             jax_deg[:, 1],
    })
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print("--- Simulation Data (in Degrees) ---")
    print(df)

    # 6. Plot
    print("\n--- Generating comparison plot ---")
    t = df['Time (s)'].values

    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Simulink vs. NumPy vs. JAX Environment Comparison', fontsize=16)

    # Angle
    axs[0].plot(t, simulink_deg[:,0], label='Simulink', linestyle='-',  linewidth=2)
    axs[0].plot(t, numpy_deg[:,0],    label='NumPy',    linestyle='--', linewidth=1.5)
    axs[0].plot(t, jax_deg[:,0],      label='JAX',      linestyle=':',  linewidth=1.5)
    axs[0].set_ylabel('θ (°)')
    axs[0].legend()
    axs[0].grid(True)

    # Angular velocity
    axs[1].plot(t, simulink_deg[:,1], label='Simulink', linestyle='-',  linewidth=2)
    axs[1].plot(t, numpy_deg[:,1],    label='NumPy',    linestyle='--', linewidth=1.5)
    axs[1].plot(t, jax_deg[:,1],      label='JAX',      linestyle=':',  linewidth=1.5)
    axs[1].set_ylabel('θ̇ (°/s)')
    axs[1].legend()
    axs[1].grid(True)

    # Differences (in radians, log scale)
    diff_np  = np.abs(np.array(simulink_history) - np.array(numpy_history))
    diff_jax = np.abs(np.array(simulink_history) - np.array(jax_history))
    axs[2].plot(t, diff_np[:,0],  label='|Sim - NumPy|', linestyle='-')
    axs[2].plot(t, diff_jax[:,0], label='|Sim - JAX|',   linestyle='--')
    axs[2].set_yscale('log')
    axs[2].set_ylabel('Abs diff (rad)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig("simulink_numpy_jax_comparison_degrees.png")
    print("Plot saved as simulink_numpy_jax_comparison_degrees.png")
    plt.show()

if __name__ == "__main__":
    run_comparison_test(seed=42, num_steps=100)
