import time
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from BCSimTestEnv import BCSimulinkEnv # Import from your uploaded environment file

class BCModelTester:
    """
    A class to test a trained Stable Baselines3 model on the BCSimulinkEnv.
    This version includes extensive debug printing for detailed analysis.
    """
    def __init__(self, model_path, stats_path, model_class=A2C):
        self.model_path = model_path
        self.stats_path = stats_path
        self.model_class = model_class
        print("Tester initialized. Use run_test_episode() to start a test.")

    def calculate_and_print_stats(self, times, voltages, target_voltage):
        """
        Calculates and prints performance statistics, including raw data for debugging.
        This version ignores the final data point to avoid end-of-episode anomalies.
        """
        if len(times) < 2:
            print("\n--- Not enough data to calculate statistics. ---")
            return

        # --- Simple Fix: Ignore the last data point from the lists ---
        times_for_stats = np.array(times[:-1])
        voltages_for_stats = np.array(voltages[:-1])
        
        # --- Statistics Calculation ---
        stabilization_time, first_stable_index = float('inf'), -1
        tolerance = 0.5
        outside_indices = np.where((voltages_for_stats < target_voltage - tolerance) | (voltages_for_stats > target_voltage + tolerance))[0]
        
        if len(outside_indices) == 0:
            first_stable_index = 0
            stabilization_time = times_for_stats[0] if len(times_for_stats) > 0 else 0
        else:
            last_unstable_index = outside_indices[-1]
            if last_unstable_index + 1 < len(times_for_stats):
                first_stable_index = last_unstable_index + 1
                stabilization_time = times_for_stats[first_stable_index]
        
        stabilization_duration = times_for_stats[-1] - stabilization_time if first_stable_index != -1 else 0
        overshoot = max(0, np.max(voltages_for_stats) - target_voltage)
        undershoot = max(0, target_voltage - np.min(voltages_for_stats))
        
        steady_state_error = None
        steady_state_errors_array = []
        if first_stable_index != -1 and len(voltages_for_stats[first_stable_index:]) > 0:
            steady_state_voltages = voltages_for_stats[first_stable_index:]
            steady_state_errors_array = steady_state_voltages - target_voltage
            steady_state_error = np.mean(steady_state_errors_array)
                
        # --- Print Results ---
        print("\n--- Performance Statistics (excluding final step) ---")
        print("-" * 50)
        if stabilization_time != float('inf'):
            print(f"Stabilization Time:   {stabilization_time * 1000:.2f} ms")
            print(f"Stabilization Duration: {stabilization_duration * 1000:.2f} ms")
        else:
            print("Stabilization Time:   Not achieved")
        print(f"Voltage Overshoot:    {overshoot:.4f} V")
        print(f"Voltage Undershoot:   {undershoot:.4f} V")
        if steady_state_error is not None:
            print(f"Avg Steady-State Err: {steady_state_error:+.4f} V")
        else:
            print("Steady-State Error:   N/A")
        print("-" * 50)
        
        # --- Debug Printing ---
        print("\n--- Full Episode Data (for debugging) ---")
        print("All Times Recorded: ")
        print(np.array(times))
        print("\nAll Voltages Recorded: ")
        print(np.array(voltages))
        if len(steady_state_errors_array) > 0:
            print("\nAll Steady-State Errors (after stabilization):")
            print(steady_state_errors_array)
        else:
            print("\nSystem did not stabilize; no steady-state errors to display.")
        print("-" * 50)


    def run_test_episode(self, target_voltage=30.0, voltage_ylim=None):
        """Runs a single test episode, displays the plot, and prints stats."""
        print(f"\n--- Starting Test for Target: {target_voltage:.2f}V ---")
        if voltage_ylim: print(f"Voltage plot zoomed to Y-limits: {voltage_ylim}")
        
        env = None
        try:
            env_fn = lambda: BCSimulinkEnv(model_name="bcSim", enable_plotting=True, target_voltage=target_voltage, voltage_plot_ylim=voltage_ylim)
            env = DummyVecEnv([env_fn])
            env = VecNormalize.load(self.stats_path, env)
            env.training = False
            env.norm_reward = False
            model = self.model_class.load(self.model_path, env=env)
            
            # --- Data Collection Loop ---
            obs = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, info = env.step(action)
                done = terminated[0] or info[0].get('TimeLimit.truncated', False)

            print("--- Test Episode Finished ---")
            
            # --- Analysis and Display ---
            # Retrieve the complete history from the environment for analysis
            all_times = env.get_attr('_times')[0]
            all_voltages = env.get_attr('_voltages')[0]
            self.calculate_and_print_stats(all_times, all_voltages, target_voltage)
            
        except FileNotFoundError as e:
            print(f"\n[ERROR] A required file was not found: {e}")
            print("Please ensure your model and stats files are in the correct location.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
        finally:
            if env:
                env.close()

if __name__ == "__main__":
    MODEL_SAVE_PATH = "a2c_bc_model_final.zip"
    STATS_PATH = "vec_normalize_stats_final.pkl"
    tester = BCModelTester(model_path=MODEL_SAVE_PATH, stats_path=STATS_PATH, model_class=A2C)
    tester.run_test_episode(target_voltage=30.0, voltage_ylim=(28.5, 31.5))
