import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import csv

# GLOBAL FONT AND STYLE CONFIGURATION
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'axes.titleweight': 'bold',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
    'legend.labelcolor': 'black',
})

def save_plot_data(filename, time_data, voltage_data):
    """Saves the time and voltage data from a simulation run to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time_s', 'voltage_v'])
        writer.writerows(zip(time_data, voltage_data))
    print(f"  Data saved to '{filename}'")

def setNoise(eng, model, noise_std_dev):
    """Sets the observational noise level in the Simulink model."""
    if noise_std_dev > 0:
        noise_variance = noise_std_dev ** 2
        eng.set_param(f'{model}/Random Number', 'Variance', str(noise_variance), nargout=0)
        # Set a new random seed for each run to ensure different noise patterns
        eng.set_param(f'{model}/Random Number', 'Seed', str(np.random.randint(1, 100000)), nargout=0)
    else:
        eng.set_param(f'{model}/Random Number', 'Variance', str(0), nargout=0)

def calculate_metrics(time_lst, voltage_lst, desired_voltage, stability_threshold):
    """Calculates performance metrics from the simulation data."""
    metrics = {
        'stabilisation_time_s': 'Did not stabilize',
        'overshoot_v': 0.0,
        'steady_state_error_v': 'N/A'
    }

    # Overshoot Calculation (adjusted for negative voltage)
    min_voltage = np.min(voltage_lst)
    if min_voltage < desired_voltage:
        metrics['overshoot_v'] = np.abs(min_voltage - desired_voltage)

    # Stabilisation Time and Steady-State Error
    stable_band_min = desired_voltage - stability_threshold
    stable_band_max = desired_voltage + stability_threshold
    
    time_delta = time_lst[1] - time_lst[0] if len(time_lst) > 1 else 0.0
    required_stable_steps = int(0.01 / time_delta) if time_delta > 0 else 0
    consecutive_stable_steps = 0
    first_stable_time = None

    for i in range(len(voltage_lst)):
        voltage = voltage_lst[i]
        if stable_band_min <= voltage <= stable_band_max:
            if consecutive_stable_steps == 0:
                first_stable_time = time_lst[i]
            consecutive_stable_steps += 1
            if required_stable_steps > 0 and consecutive_stable_steps >= required_stable_steps:
                metrics['stabilisation_time_s'] = first_stable_time
                settled_region = voltage_lst[int(len(voltage_lst) * 0.8):]
                metrics['steady_state_error_v'] = np.mean(np.abs(np.array(settled_region) - desired_voltage))
                break
        else:
            consecutive_stable_steps = 0
            first_stable_time = None
            
    return metrics

def main(noise_levels_to_test = [0.0, 0.001, 0.01, 0.1],
         goal_voltages = [-30.0, -80.0, -110.0],
         model = 'bbcSimPID'):
    """
    Method to set up MATLAB, run simulations for multiple episodes, and generate stylized plots.
    """
    print("Setting up MATLAB engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)

    colors = {-30.0: '#1f77b4', -80.0: '#ff7f0e', -110.0: 'green'}

    for noise_level in noise_levels_to_test:
        print(f"\n--- TESTING NOISE LEVEL σ = {noise_level} ---")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.2)

        for i, goal in enumerate(goal_voltages):
            # Define and plot error bounds once per goal voltage
            error_bound = abs(0.04 * goal)
            label = '±4% Error Bound' if i == 0 else ""
            ax.axhline(y=goal + error_bound, color='k', linestyle='--', linewidth=1.5, label=label)
            ax.axhline(y=goal - error_bound, color='k', linestyle='--', linewidth=1.5)

            # Dictionary to store metrics from all episodes for this goal
            all_episode_metrics = {
                'stabilisation_time_s': [],
                'overshoot_v': [],
                'steady_state_error_v': []
            }

            # Run 3 episodes for each voltage goal
            for episode in range(3):
                print(f"\nRunning simulation for Goal: {goal}V, Episode: {episode + 1}/3...")
                eng.set_param(f'{model}/finalVoltage', 'Value', str(goal), nargout=0)
                eng.py.PIDControllerBBC.reset_controller(nargout=0)
                setNoise(eng, model, noise_level)
                eng.eval(f"out = sim('{model}');", nargout=0)
                
                voltage_2d = eng.eval("out.voltage")
                voltage_lst = [v[0] for v in voltage_2d]
                time_2d = eng.eval("out.tout")
                time_lst = [t[0] for t in time_2d]
                print("  Simulation complete.")

                metrics = calculate_metrics(time_lst, voltage_lst, goal, stability_threshold=error_bound)
                
                # Store metrics for later averaging
                all_episode_metrics['stabilisation_time_s'].append(metrics['stabilisation_time_s'])
                all_episode_metrics['overshoot_v'].append(metrics['overshoot_v'])
                all_episode_metrics['steady_state_error_v'].append(metrics['steady_state_error_v'])

                print(f"  Performance Metrics (Episode {episode + 1}):")
                if isinstance(metrics['stabilisation_time_s'], float):
                    print(f"    - Stabilisation Time: {metrics['stabilisation_time_s']:.4f} s")
                else:
                    print(f"    - Stabilisation Time: {metrics['stabilisation_time_s']}")
                print(f"    - Overshoot: {metrics['overshoot_v']:.2f} V")
                if isinstance(metrics['steady_state_error_v'], float):
                    print(f"    - Steady-State Error: {metrics['steady_state_error_v']:.4f} V")
                else:
                     print(f"    - Steady-State Error: {metrics['steady_state_error_v']}")

                # Save data for each individual episode with a unique filename
                plot_filename = f"plot_data_bbc_noise_{noise_level}_goal_{abs(goal)}V_ep_{episode+1}.csv"
                save_plot_data(plot_filename, time_lst, voltage_lst)

                # Plot the result, but only label the first episode for each goal to keep the legend clean
                plot_label = f'Goal: {goal}V' if episode == 0 else ""
                ax.plot(time_lst, voltage_lst, label=plot_label, linewidth=1.2, color=colors.get(goal))
            
            # --- Calculate and print average metrics after all episodes for this goal ---
            print(f"\n  Average Metrics for Goal: {goal}V (over 3 episodes):")
            
            # Stabilisation Time
            valid_stab_times = [t for t in all_episode_metrics['stabilisation_time_s'] if isinstance(t, float)]
            if valid_stab_times:
                avg_stab_time = np.mean(valid_stab_times)
                std_stab_time = np.std(valid_stab_times)
                print(f"    - Avg. Stabilisation Time: {avg_stab_time:.4f} ± {std_stab_time:.4f} s ({len(valid_stab_times)}/3 runs stabilized)")
            else:
                print("    - Avg. Stabilisation Time: Did not stabilize in any run")

            # Overshoot
            overshoots = all_episode_metrics['overshoot_v']
            avg_overshoot = np.mean(overshoots)
            std_overshoot = np.std(overshoots)
            print(f"    - Avg. Overshoot: {avg_overshoot:.2f} ± {std_overshoot:.2f} V")

            # Steady-State Error
            valid_ss_errors = [e for e in all_episode_metrics['steady_state_error_v'] if isinstance(e, float)]
            if valid_ss_errors:
                avg_ss_error = np.mean(valid_ss_errors)
                std_ss_error = np.std(valid_ss_errors)
                print(f"    - Avg. Steady-State Error: {avg_ss_error:.4f} ± {std_ss_error:.4f} V")
            else:
                print("    - Avg. Steady-State Error: N/A (Could not determine from runs)")


        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        
        ax.set_xlim(0, 0.07)
        
        ax.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='lightgrey')
        
        ax.grid(False)

        output_filename = f"bbc_performance_noise_{noise_level}`_0.07`.svg"
        plt.tight_layout()
        plt.savefig(output_filename, format='svg', bbox_inches='tight')
        print(f"\n📈 Plot saved successfully as '{output_filename}'")
        plt.close()

    print("\nAll simulations finished. Quitting MATLAB engine.")
    eng.quit()

if __name__ == '__main__':
    main()

