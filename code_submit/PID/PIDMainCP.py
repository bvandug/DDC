# PIDMainCP.py

import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import numbers

# Global Font and Style Configuration
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

def set_noise(eng, model, noise_std_dev):
    """Sets the observational noise level in the Simulink model.

    Args:
        eng (matlab.engine.MatlabEngine): The active MATLAB engine instance.
        model (str): The name of the Simulink model.
        noise_std_dev (float): The standard deviation of the desired noise.

    Returns:
        None
    """
    noise_block_path = f'{model}/Random Number'
    if noise_std_dev > 0:
        noise_variance = noise_std_dev ** 2
        eng.set_param(noise_block_path, 'Variance', str(noise_variance), nargout=0)
        eng.set_param(noise_block_path, 'Seed', str(np.random.randint(1, 100000)), nargout=0)
        print(f"  Noise enabled with standard deviation {noise_std_dev} (Variance: {noise_variance})")
    else:
        eng.set_param(noise_block_path, 'Variance', str(0), nargout=0)
        print("  Noise disabled.")

def calculate_metrics(time_lst, angle_lst, stability_threshold=0.1):
    """Calculates performance metrics for a single simulation run.

    Args:
        time_lst (list): List of timestamps from the simulation.
        angle_lst (list): List of angle measurements from the simulation.
        stability_threshold (float): The error margin to define stability.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    metrics = {'stabilisation_time_s': 'Did not stabilize', 'overshoot_rad': 0.0, 'steady_state_error_rad': 'N/A'}
    if not angle_lst or len(angle_lst) < 2: return metrics
    start_angle = angle_lst[0]; start_sign = np.sign(start_angle)
    first_crossing_index = -1
    for i in range(1, len(angle_lst)):
        if np.sign(angle_lst[i]) != start_sign and np.sign(angle_lst[i]) != 0:
            first_crossing_index = i; break
    if first_crossing_index != -1:
        overshoot_region = angle_lst[first_crossing_index:]
        if overshoot_region:
            if start_sign > 0: metrics['overshoot_rad'] = abs(min(overshoot_region)) if any(x < 0 for x in overshoot_region) else 0.0
            else: metrics['overshoot_rad'] = max(overshoot_region) if any(x > 0 for x in overshoot_region) else 0.0
    simulation_time = 3.0
    time_delta = simulation_time / len(time_lst) if len(time_lst) > 1 else 0
    if time_delta == 0: return metrics
    required_stable_steps = int(1.0 / time_delta); consecutive_stable_steps = 0
    first_stable_time = None; first_stable_index = -1
    for i in range(len(angle_lst)):
        if abs(angle_lst[i]) <= stability_threshold:
            if first_stable_time is None: first_stable_time = time_lst[i]; first_stable_index = i
            consecutive_stable_steps += 1
            if consecutive_stable_steps >= required_stable_steps:
                metrics['stabilisation_time_s'] = first_stable_time
                settled_angles = angle_lst[first_stable_index:]
                if settled_angles: metrics['steady_state_error_rad'] = np.mean(np.abs(settled_angles))
                break
        else:
            consecutive_stable_steps = 0; first_stable_time = None; first_stable_index = -1
    return metrics

def main(model='pendSimPID', mask='Pendulum and Cart', simulation_time=3.0, stability_threshold=0.1):
    """Main function to run the cart-pole simulation experiment.

    Args:
        model (str): The name of the Simulink model file.
        mask (str): The name of the masked subsystem for setting parameters.
        simulation_time (float): The total simulation time in seconds.
        stability_threshold (float): Error margin for metrics and plotting.

    Returns:
        None
    """
    # Simulation Parameters
    noise_levels_to_test = [0.0, 0.001, 0.01, 0.1]
    initial_angles_to_test = [0.455, -0.944, 0.156, -0.620, -0.842]
    
    # MATLAB Engine Setup
    print("Setting up MATLAB engine..."); eng = matlab.engine.start_matlab()
    eng.addpath('.', nargout=0); eng.load_system(model, nargout=0)
    eng.set_param(model, 'StopTime', str(simulation_time), nargout=0)
    all_results = {}; plt.style.use('seaborn-v0_8-whitegrid')

    # Primary Loop: Iterate over each noise level
    for noise_level in noise_levels_to_test:
        print(f"\n{'='*60}\n===== TESTING NOISE LEVEL σ = {noise_level} =====\n{'='*60}")
        all_results[noise_level] = {}; summary_plot_data = []

        # Secondary Loop: Iterate over each initial angle
        for initial_angle in initial_angles_to_test:
            print(f"\n--- Testing Initial Angle = {initial_angle:.4f} rad ---")
            eng.set_param(f'{model}/{mask}', 'init', str(initial_angle), nargout=0)
            
            # Run Simulation for this configuration
            print("  Running simulation with CLAMPED controller...")
            eng.py.PIDControllerCP.reset_controller(nargout=0)
            eng.py.PIDControllerCP.set_clamping(True, nargout=0)
            set_noise(eng, model, noise_level)
            eng.eval(f"out = sim('{model}');", nargout=0)

            # Extract and process simulation data
            angle_data, time_data = eng.eval("out.angle"), eng.eval("out.tout")
            if isinstance(angle_data, numbers.Number): angle_lst, time_lst = [angle_data], [time_data]
            else: angle_lst, time_lst = [i[0] for i in angle_data], [i[0] for i in time_data]
            
            # Calculate and store metrics
            metrics = calculate_metrics(time_lst, angle_lst, stability_threshold)
            all_results[noise_level][initial_angle] = metrics
            if time_lst: summary_plot_data.append({'angle': initial_angle, 'data': (time_lst, angle_lst)})
            print("  Simulation complete.")

        # Summary Plot Generation (per noise level)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.spines['top'].set_color('black'); ax.spines['bottom'].set_color('black'); ax.spines['left'].set_color('black'); ax.spines['right'].set_color('black')
        ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.2)
        for item in summary_plot_data:
            time, angle = item['data']
            ax.plot(time, angle, label=f'Start Angle: {item["angle"]:.3f} rad', linewidth=1.5)
        ax.axhline(y=stability_threshold, color='k', linestyle='--', label=f'±{stability_threshold} rad Error Bound')
        ax.axhline(y=-stability_threshold, color='k', linestyle='--')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (rad)")
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='lightgrey')
        ax.grid(False); ax.set_xlim(0, simulation_time)
        summary_plot_filename = f'CP_clamped_summary_noise_{noise_level}.svg'
        plt.tight_layout(); plt.savefig(summary_plot_filename, bbox_inches='tight'); plt.close(fig)
        print(f"\nSummary plot saved as '{summary_plot_filename}'")

    eng.quit()
    
    # Final Summary Table Generation
    print("\n\n" + "="*80 + "\n||" + " "*21 + "FINAL CLAMPED PERFORMANCE SUMMARY" + " "*22 + "||\n" + "="*80)
    for noise_level, angle_data in all_results.items():
        print(f"\n--- Performance at Noise Level σ = {noise_level} ---")
        print("| Initial Angle (rad) | Stabilisation Time (s) | Overshoot (rad) | Steady-State Error (rad) |")
        print("|:--------------------|:-----------------------|:----------------|:-------------------------|")
        for angle, metrics in angle_data.items():
            stab_str = f"{metrics['stabilisation_time_s']:.3f}" if isinstance(metrics['stabilisation_time_s'], numbers.Number) else "N/A"
            over_str = f"{metrics['overshoot_rad']:.3f}"
            sse_str = f"{metrics['steady_state_error_rad']:.4f}" if isinstance(metrics['steady_state_error_rad'], numbers.Number) else "N/A"
            print(f"| {angle:<19.4f} | {stab_str:<22} | {over_str:<15} | {sse_str:<24} |")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()