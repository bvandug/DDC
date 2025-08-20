import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

def setNoise(eng, model, noise_std_dev):
    """
    Sets the observational noise level in the Simulink model's Random Number block.
    NOTE: The variance is the square of the standard deviation.
    """
    # Corrected block path to use 'Random Number' as you specified.
    noise_block_path = f'{model}/Random Number'
    if noise_std_dev > 0:
        noise_variance = noise_std_dev ** 2
        # Corrected parameter from 'Cov' to 'Variance' for the Random Number block.
        eng.set_param(noise_block_path, 'Variance', str(noise_variance), nargout=0)
        eng.set_param(noise_block_path, 'Seed', str(np.random.randint(1, 100000)), nargout=0)
        print(f"  Noise enabled with standard deviation {noise_std_dev} (Variance: {noise_variance})")
    else:
        eng.set_param(noise_block_path, 'Variance', str(0), nargout=0)
        print("  Noise disabled.")

def calculate_metrics(time_lst, angle_lst, stability_threshold=0.1):
    """
    Calculates performance metrics for the pendulum simulation.
    """
    metrics = {
        'stabilisation_time_s': 'Did not stabilize',
        'overshoot_rad': 0.0,
        'steady_state_error_rad': 'N/A'
    }

    # --- Overshoot Calculation (absolute deviation from 0) ---
    if angle_lst:
        metrics['overshoot_rad'] = np.max(np.abs(angle_lst))

    # --- Stabilisation Time and Steady-State Error Calculation ---
    stable_band = stability_threshold
    
    # Require stability for 1 second
    if not time_lst or len(time_lst) < 2:
        return metrics # Cannot calculate if time is empty or has one point

    time_delta = time_lst[1] - time_lst[0]
    if time_delta == 0:
        return metrics # Avoid division by zero

    required_stable_steps = int(1.0 / time_delta)
    consecutive_stable_steps = 0
    first_stable_time = None

    for i in range(len(angle_lst)):
        angle = angle_lst[i]
        if abs(angle) <= stable_band:
            if first_stable_time is None:
                first_stable_time = time_lst[i]
            consecutive_stable_steps += 1
            if consecutive_stable_steps >= required_stable_steps:
                metrics['stabilisation_time_s'] = first_stable_time
                
                # Calculate steady-state error on the last 20% of the data
                settled_region_start_index = int(len(angle_lst) * 0.8)
                settled_region = angle_lst[settled_region_start_index:]
                if settled_region:
                    metrics['steady_state_error_rad'] = np.mean(np.abs(np.array(settled_region)))
                break
        else:
            consecutive_stable_steps = 0
            first_stable_time = None
            
    return metrics

def main(model = 'pendSimPID',
         mask = 'Pendulum and Cart',
         stabilisation_precision = 0.1,
         initial_angle = np.pi/3):
    """
    Method to set up MATLAB, Simulink, and handle data acquisition,
    metric calculation, and plotting for the pendulum system.
    """
    noise_levels_to_test = [0.0, 0.001, 0.01, 0.1]
    num_episodes = 3

    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)
    
    # Setting fixed model parameters
    eng.set_param(f'{model}/{mask}', 'init', str(initial_angle), nargout=0)

    for noise_level in noise_levels_to_test:
        print(f"\n--- Testing Noise Level σ = {noise_level} ---")
        
        all_episode_metrics = []
        episode_results = []
        
        for episode in range(1, num_episodes + 1):
            print(f"  Running episode {episode}/{num_episodes}...")
            
            # --- Reset the Python controller before each run ---
            # This calls the new reset_controller function in the other file
            eng.py.PIDControllerPend.reset_controller(nargout=0)
            #print("  PID controller state has been reset.")
            
            setNoise(eng, model, noise_level)

            # Run the simulation
            eng.eval(f"out = sim('{model}');", nargout=0)
            
            # Get angles
            angle_2d = eng.eval("out.angle")
            angle_lst = [angle[0] for angle in angle_2d]

            # Get time
            time_2d = eng.eval("out.tout")
            time_lst = [time[0] for time in time_2d]
            
            # Calculate and store metrics
            metrics = calculate_metrics(time_lst, angle_lst)
            all_episode_metrics.append(metrics)
            episode_results.append((time_lst, angle_lst))

            # --- Print Metrics for the Current Episode ---
            print("    Performance Metrics for this episode:")
            stab_time = metrics['stabilisation_time_s']
            stab_time_str = f"{stab_time:.2f} s" if isinstance(stab_time, (int, float)) else stab_time
            print(f"      - Stabilisation Time: {stab_time_str}")

            overshoot = metrics['overshoot_rad']
            print(f"      - Overshoot: {overshoot:.2f} rad")

            sse = metrics['steady_state_error_rad']
            sse_str = f"{sse:.4f} rad" if isinstance(sse, (int, float)) else sse
            print(f"      - Steady-State Error: {sse_str}\n")

        # --- Calculate and Print Average Metrics ---
        valid_stab_times = [m['stabilisation_time_s'] for m in all_episode_metrics if isinstance(m['stabilisation_time_s'], (int, float))]
        avg_stab_time = np.mean(valid_stab_times) if valid_stab_times else 'N/A'
        std_stab_time = np.std(valid_stab_times) if valid_stab_times else 'N/A'
        
        avg_overshoot = np.mean([m['overshoot_rad'] for m in all_episode_metrics])
        std_overshoot = np.std([m['overshoot_rad'] for m in all_episode_metrics])
        
        valid_sse = [m['steady_state_error_rad'] for m in all_episode_metrics if isinstance(m['steady_state_error_rad'], (int, float))]
        avg_sse = np.mean(valid_sse) if valid_sse else 'N/A'
        std_sse = np.std(valid_sse) if valid_sse else 'N/A'

        # --- FIXED PRINTING LOGIC ---
        print("\n  Average Performance Metrics:")
        
        # Only format as a float if it's a number, otherwise print the string ('N/A')
        stab_time_str = f"{avg_stab_time:.2f} ± {std_stab_time:.2f}" if isinstance(avg_stab_time, (int, float)) else avg_stab_time
        print(f"    - Avg. Stabilisation Time: {stab_time_str} s")

        overshoot_str = f"{avg_overshoot:.2f} ± {std_overshoot:.2f}"
        print(f"    - Avg. Overshoot: {overshoot_str} rad")

        sse_str = f"{avg_sse:.4f} ± {std_sse:.4f}" if isinstance(avg_sse, (int, float)) else avg_sse
        print(f"    - Avg. Steady-State Error: {sse_str} rad")
        
        # --- Create and Save the Plot ---
        plt.figure(figsize=(12, 8))
        for i in range(num_episodes):
            time, angle = episode_results[i]
            plt.plot(time, angle, label=f'Run {i+1}')
        
        plt.axhline(y=stabilisation_precision, color='k', linestyle='--', label=f'±{stabilisation_precision} rad Error Bound')
        plt.axhline(y=-stabilisation_precision, color='k', linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad)")
        plt.title(f"PID Controller Performance on Pendulum with Noise σ = {noise_level} (3 Runs)")
        plt.legend()
        plt.grid(True)
        if time_lst:
            plt.xlim(0, max(time_lst))
        
        plot_filename = f'PID_pendulum_performance_noise_{noise_level}.svg'
        plt.savefig(plot_filename)
        print(f"Plot saved as '{plot_filename}'")
        plt.close()

    eng.quit()

if __name__ == '__main__':
    main()
