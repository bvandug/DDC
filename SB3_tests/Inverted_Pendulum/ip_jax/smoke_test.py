import numpy as np
import jax
import matplotlib.pyplot as plt

# Enforce 64-bit precision for JAX to match NumPy
jax.config.update("jax_enable_x64", True)

# Import the environment wrappers from your files
from ip_numpy_wrapper import InvertedPendulumGymWrapper as NumpyEnvWrapper
from ip_jax_wrapper import InvertedPendulumGymWrapper as JaxEnvWrapper
# Import the state definition from one of the envs to manually create a state
from ip_jax import PendulumState

def run_smoke_test(seed=42, num_steps=500, tolerance=0.1):
    """
    Initializes both NumPy and JAX environments with the same seed,
    applies the same sequence of actions, compares their states at each step,
    and plots the results.
    """
    print("--- Starting Smoke Test: Comparing NumPy and JAX Environments ---")
    print(f"--- JAX is configured to use 64-bit precision to match NumPy. ---")
    print(f"--- Using a tolerance of {tolerance} to account for minor floating-point differences. ---")


    # 1. Initialize both environments
    print(f"Initializing both environments with seed: {seed}")
    numpy_env = NumpyEnvWrapper(seed=seed)
    jax_env = JaxEnvWrapper(seed=seed)

    # 2. Reset NumPy env to get a valid random starting state
    numpy_obs, _ = numpy_env.reset(seed=seed)
    print(f"NumPy environment started at: {numpy_obs}")

    # 3. Force the JAX environment to start from the exact same state
    jax_env.state = PendulumState(
        theta=numpy_obs[0],
        theta_dot=numpy_obs[1],
        t=0.0,
        done=False
    )
    jax_obs = jax_env._obs()
    print(f"JAX environment manually set to: {jax_obs}")

    # Lists to store history for plotting
    numpy_history = [numpy_obs]
    jax_history = [jax_obs]
    diverged = False

    # Check initial state
    if not np.allclose(numpy_obs, jax_obs, atol=tolerance):
        print("\n❌ Failure: States could not be synchronized!")
        return
    else:
        print("Initial states synchronized. Proceeding with action sequence.")

    # 4. Define a deterministic sequence of actions to apply
    action_sequence = [np.array([np.sin(i * 0.1) * 2.0], dtype=np.float32) for i in range(num_steps)]

    # 5. Loop through the actions and compare states
    for i, action in enumerate(action_sequence):
        numpy_obs, _, _, _, _ = numpy_env.step(action)
        jax_obs, _, _, _, _ = jax_env.step(action)

        # Store history for plotting
        numpy_history.append(numpy_obs)
        jax_history.append(jax_obs)

        # Check for divergence but don't stop the loop
        if not np.allclose(numpy_obs, jax_obs, atol=tolerance) and not diverged:
            diverged = True
            print(f"\n⚠️ First state divergence detected at step {i + 1}!")
            print(f"  Action taken: {action[0]:.4f}")
            print(f"  NumPy obs: {numpy_obs}")
            print(f"  JAX obs:   {jax_obs}")
            print(f"  Difference: {np.abs(numpy_obs - jax_obs)}")
            print("  Continuing run to plot full divergence...")

    if not diverged:
        print(f"\n✅ Success! Both environments produced functionally identical states for {num_steps} steps.")
    else:
        print("\n❌ Failure! The environments diverged. Check the plot for details.")

    numpy_env.close()
    jax_env.close()

    # 6. Plot the results
    print("\n--- Generating comparison plot ---")
    numpy_history = np.array(numpy_history)
    jax_history = np.array(jax_history)
    time_steps = np.arange(num_steps + 1) * numpy_env.config.dt

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('NumPy vs. JAX Environment Smoke Test Comparison', fontsize=16)

    # Plot Theta (Angle)
    axs[0].plot(time_steps, numpy_history[:, 0], label='NumPy Theta', color='blue')
    axs[0].plot(time_steps, jax_history[:, 0], label='JAX Theta', color='red', linestyle='--')
    axs[0].set_ylabel('Angle (rad)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Theta_dot (Angular Velocity)
    axs[1].plot(time_steps, numpy_history[:, 1], label='NumPy Theta_dot', color='blue')
    axs[1].plot(time_steps, jax_history[:, 1], label='JAX Theta_dot', color='red', linestyle='--')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot the difference
    difference = np.abs(numpy_history - jax_history)
    axs[2].plot(time_steps, difference[:, 0], label='Theta Difference (abs)', color='green')
    axs[2].plot(time_steps, difference[:, 1], label='Theta_dot Difference (abs)', color='purple', linestyle=':')
    axs[2].set_ylabel('Absolute Difference')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_yscale('log') # Use a log scale to see small differences
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = "smoke_test_comparison.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show() # This can be commented out if running in a non-interactive environment

if __name__ == "__main__":
    run_smoke_test(num_steps=500)
