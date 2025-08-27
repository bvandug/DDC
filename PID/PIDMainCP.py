# PIDMainCP.py

import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

# --- Global Font and Style Configuration ---
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
    'legend.labelcolor': 'black',
})

def setNoise(eng, model, noise_std_dev):
    """
    Sets the observational noise level in the Simulink model's Random Number block.
    NOTE: The variance is the square of the standard deviation.
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

def main(model = 'pendSimPID',
         mask = 'Pendulum and Cart',
         stabilisation_precision = 0.1,
         simulation_time = 3.0):
    """
    Method to compare clamped vs. unclamped PID controller performance
    for the pendulum system.
    """
    initial_angle = -0.944298505783081 # Set initial angle to 1 radian
    noise_level = 0      # No observational noise

    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)
    
    print(f"Setting simulation StopTime to {simulation_time}s")
    eng.set_param(model, 'StopTime', str(simulation_time), nargout=0)
    
    # --- Set shared simulation parameters ---
    print(f"\n--- Testing Initial Angle: {initial_angle:.4f} rad ---")
    eng.set_param(f'{model}/{mask}', 'init', str(initial_angle), nargout=0)
    setNoise(eng, model, noise_level) # Disables noise

    # === RUN 1: CLAMPED CONTROLLER ===
    print("\nRunning simulation with CLAMPED controller...")
    eng.py.PIDControllerCP.reset_controller(nargout=0)
    eng.py.PIDControllerCP.set_clamping(True, nargout=0) # Ensure clamping is ON
    eng.eval(f"out_clamped = sim('{model}');", nargout=0)
    
    # Extract clamped data
    angle_clamped = [angle[0] for angle in eng.eval("out_clamped.angle")]
    time_clamped = [time[0] for time in eng.eval("out_clamped.tout")]
    print("  Clamped simulation complete.")

    # === RUN 2: UNCLAMPED CONTROLLER ===
    print("\nRunning simulation with UNCLAMPED controller...")
    eng.py.PIDControllerCP.reset_controller(nargout=0)
    eng.py.PIDControllerCP.set_clamping(False, nargout=0) # Turn clamping OFF
    eng.eval(f"out_unclamped = sim('{model}');", nargout=0)

    # Extract unclamped data
    angle_unclamped = [angle[0] for angle in eng.eval("out_unclamped.angle")]
    time_unclamped = [time[0] for time in eng.eval("out_unclamped.tout")]
    print("  Unclamped simulation complete.")

    eng.quit()

    # === PLOTTING (MATCHING ORIGINAL FORMAT) ===
    print("\nGenerating comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Setup plot aesthetics from original file
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.2)

    # Plot both results
    ax.plot(time_clamped, angle_clamped, label='Clamped Output', linewidth=1.5, color = 'r')
    ax.plot(time_unclamped, angle_unclamped, label='Unclamped Output', linewidth=1.5, color = 'g')
    
    # Plot error bounds
    ax.axhline(y=stabilisation_precision, color='k', linestyle='--', linewidth=1.5, label=f'±{stabilisation_precision} rad Error Bound')
    ax.axhline(y=-stabilisation_precision, color='k', linestyle='--', linewidth=1.5)

    # Set labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    
    if time_clamped:
        ax.set_xlim(left=0, right=max(time_clamped))
    
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='lightgrey')
    ax.grid(False)
    
    plot_filename = 'PID_clamped_vs_unclamped_comparison.svg'
    plt.tight_layout()
    plt.savefig(plot_filename, format='svg', bbox_inches='tight')
    print(f"Comparison plot saved as '{plot_filename}'")
    plt.close(fig)

if __name__ == '__main__':
    main()