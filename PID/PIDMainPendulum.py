# PIDMainPendulum.py (Corrected with Fixed X-Axis)

import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import numbers # Import the numbers module to check for float/int

# --- Global Font and Style Configuration ---
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
    """Sets the observational noise level in the Simulink model."""
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
    """
    Calculates performance metrics for a single simulation run.
    """
    metrics = {
        'stabilisation_time_s': 'Did not stabilize', 'overshoot_rad': 0.0, 'steady_state_error_rad': 'N/A'
    }
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
    # Set simulation time to 3 seconds for metric calculation consistency
    simulation_time = 1.5
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

def run_simulation():
    """Main function to run the pendulum simulation experiment."""
    model_name = 'pendulumSimPID'; mask_name = 'Pendulum'
    simulation_time = 1.5 # Define simulation time
    noise_levels_to_test = [0.0, 0.001, 0.01, 0.1]
    initial_angles_to_test = [
        0.45532846450805664, -0.944298505783081, 0.1558818817138672,
        -0.6195435523986816, -0.8419008255004883
    ]
    num_episodes_stochastic = 1; enable_clipping = True; action_limit = 2.0
    print("Setting up MATLAB engine..."); eng = matlab.engine.start_matlab()
    eng.addpath('.', nargout=0); eng.load_system(model_name, nargout=0)
    # Set simulation StopTime in Simulink
    eng.set_param(model_name, 'StopTime', str(simulation_time), nargout=0)
    all_results = {}; plt.style.use('seaborn-v0_8-whitegrid')

    for noise_level in noise_levels_to_test:
        print(f"\n{'='*60}\n===== TESTING NOISE LEVEL σ = {noise_level} =====\n{'='*60}")
        all_results[noise_level] = {}; summary_plot_data = []

        for initial_angle in initial_angles_to_test:
            print(f"\n--- Testing Initial Angle = {initial_angle:.4f} rad ---")
            eng.set_param(f'{model_name}/{mask_name}', 'init', str(initial_angle), nargout=0)
            num_episodes = 1 if noise_level == 0.0 else num_episodes_stochastic
            episode_metrics, episode_plots = [], []
            
            for episode in range(1, num_episodes + 1):
                print(f"  Running episode {episode}/{num_episodes}...")
                eng.py.PIDControllerPendulum.configure_controller_clipping(enable_clipping, action_limit, nargout=0)
                eng.py.PIDControllerPendulum.reset_controller(nargout=0)
                set_noise(eng, model_name, noise_level)
                eng.eval(f"out = sim('{model_name}');", nargout=0)
                angle_data, time_data = eng.eval("out.angle"), eng.eval("out.tout")
                if isinstance(angle_data, numbers.Number): angle_lst, time_lst = [angle_data], [time_data]
                else: angle_lst, time_lst = [i[0] for i in angle_data], [i[0] for i in time_data]
                episode_metrics.append(calculate_metrics(time_lst, angle_lst)); episode_plots.append((time_lst, angle_lst))
                
            stab_times = [m['stabilisation_time_s'] for m in episode_metrics if isinstance(m['stabilisation_time_s'], numbers.Number)]
            overshoots = [m['overshoot_rad'] for m in episode_metrics]
            sses = [m['steady_state_error_rad'] for m in episode_metrics if isinstance(m['steady_state_error_rad'], numbers.Number)]
            avg_metrics = {
                'stab_time_mean': np.mean(stab_times) if stab_times else 'N/A', 'stab_time_std': np.std(stab_times) if stab_times else 0,
                'overshoot_mean': np.mean(overshoots), 'overshoot_std': np.std(overshoots),
                'sse_mean': np.mean(sses) if sses else 'N/A', 'sse_std': np.std(sses) if sses else 0
            }
            all_results[noise_level][initial_angle] = avg_metrics
            if episode_plots: summary_plot_data.append({'angle': initial_angle, 'data': episode_plots[0]})

            # --- INDIVIDUAL PLOT ---
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.spines['top'].set_color('black'); ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black'); ax.spines['right'].set_color('black')
            ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.2)
            
            for i, (time, angle) in enumerate(episode_plots):
                ax.plot(time, angle, label=f'Run {i+1}', linewidth=1.5)
            
            ax.axhline(y=0.1, color='k', linestyle='--', label='±0.1 rad Error Bound')
            ax.axhline(y=-0.1, color='k', linestyle='--')
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (rad)")
            clipping_status = f"Clipped [{action_limit}N]" if enable_clipping else "Unclipped"
            #ax.set_title(f"PID Performance (θ₀={initial_angle:.4f} rad, σ={noise_level}, {clipping_status})")
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='lightgrey')
            ax.grid(False)
            ax.set_xlim(0, 1.5) # Set x-axis from 0 to 3 seconds
            
            plot_filename = f'pendulum_angle_{initial_angle:.4f}_noise_{noise_level}.svg'
            plt.tight_layout(); plt.savefig(plot_filename, bbox_inches='tight'); plt.close(fig)
            print(f"  Individual plot saved as '{plot_filename}'")
        
        # --- SUMMARY PLOT ---
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.spines['top'].set_color('black'); ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black'); ax.spines['right'].set_color('black')
        ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.2)

        for item in summary_plot_data:
            time, angle = item['data']
            ax.plot(time, angle, label=f'Start Angle: {item["angle"]:.3f} rad', linewidth=1.5)
            
        ax.axhline(y=0.1, color='k', linestyle='--', label='±0.1 rad Error Bound')
        ax.axhline(y=-0.1, color='k', linestyle='--')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (rad)")
        #ax.set_title(f"PID Performance Across Angles (Noise σ = {noise_level})")
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='lightgrey')
        ax.grid(False)
        ax.set_xlim(0, 1.5) # Set x-axis from 0 to 3 seconds
        
        summary_plot_filename = f'pendulum_summary_noise_{noise_level}.svg'
        plt.tight_layout(); plt.savefig(summary_plot_filename, bbox_inches='tight'); plt.close(fig)
        print(f"\nSummary plot saved as '{summary_plot_filename}'")

    eng.quit()
    
    # --- FINAL SUMMARY TABLES ---
    print("\n\n" + "="*80)
    print("||" + " "*24 + "FINAL PERFORMANCE SUMMARY" + " "*25 + "||")
    print("="*80)
    for noise_level, angle_data in all_results.items():
        print(f"\n--- Performance at Noise Level σ = {noise_level} ---")
        print("| Initial Angle (rad) | Avg. Stabilisation Time (s) | Avg. Overshoot (rad) | Avg. Steady-State Error (rad) |")
        print("|:--------------------|:----------------------------|:---------------------|:------------------------------|")
        for angle, metrics in angle_data.items():
            stab_str = f"{metrics['stab_time_mean']:.3f} ± {metrics['stab_time_std']:.3f}" if isinstance(metrics['stab_time_mean'], numbers.Number) else "N/A"
            over_str = f"{metrics['overshoot_mean']:.3f} ± {metrics['overshoot_std']:.3f}"
            sse_str = f"{metrics['sse_mean']:.4f} ± {metrics['sse_std']:.4f}" if isinstance(metrics['sse_mean'], numbers.Number) else "N/A"
            print(f"| {angle:<19.4f} | {stab_str:<27} | {over_str:<20} | {sse_str:<29} |")
    print("\n" + "="*80)

if __name__ == '__main__':
    run_simulation()