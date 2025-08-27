import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- Global Font and Style Configuration ---
# This ensures the plot has the specified professional appearance.
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

def plot_combined_performance(rl_file_path, pid_file_path, goal_voltage=30.0, error_margin=0.5):
    """
    Reads performance data from two CSV files (one for RL models, one for PID)
    and generates a single, comparative plot with a zoomed-in inset.

    Args:
        rl_file_path (str): Path to the CSV file with RL model data (e.g., A2C, SAC).
        pid_file_path (str): Path to the CSV data file for the PID controller.
        goal_voltage (float): The target voltage for the controller.
        error_margin (float): The acceptable voltage error margin.
    """
    # --- Load Data from RL Controllers CSV File ---
    try:
        df_rl = pd.read_csv(rl_file_path)
        print(f"✅ Successfully loaded the RL data file: '{rl_file_path}'")
    except FileNotFoundError:
        print(f"❌ Error: The file '{rl_file_path}' was not found.")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the RL CSV file: {e}")
        return

    # --- Load Data from PID Controller CSV File ---
    try:
        df_pid = pd.read_csv(pid_file_path)
        print(f"✅ Successfully loaded the PID data file: '{pid_file_path}'")
    except FileNotFoundError:
        print(f"❌ Error: The PID file '{pid_file_path}' was not found.")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the PID CSV file: {e}")
        return

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Set a more pronounced border for the main plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
        
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.2)

    # --- Plot RL Algorithm Data ---
    rl_colors = {'A2C': '#1f77b4', 'SAC': '#ff7f0e'} 
    for algo in ['A2C', 'SAC']:
        if algo in df_rl.columns:
            ax.plot(df_rl['Time (s)'], df_rl[algo], label=f'{algo} Controller', linewidth=1.2, color=rl_colors.get(algo))
        else:
            print(f"⚠️ Warning: Column for algorithm '{algo}' not found. It will be skipped.")

    # --- Plot PID Controller Data ---
    if 'time_s' in df_pid.columns and 'voltage_v' in df_pid.columns:
        ax.plot(df_pid['time_s'], df_pid['voltage_v'], label='PID Controller', linewidth=1.2, color='green')
    else:
        print("⚠️ Warning: PID CSV must contain 'time_s' and 'voltage_v' columns. Skipping PID plot.")

    # --- Plot Goal and Error Bounds ---
    upper_bound = goal_voltage + error_margin
    lower_bound = goal_voltage - error_margin
    ax.axhline(y=upper_bound, color='k', linestyle='--', linewidth=1.5, label=f'±{error_margin}V Error Bound')
    ax.axhline(y=lower_bound, color='k', linestyle='--', linewidth=1.5)

    # --- Final Touches and Labels for Main Plot ---
    #ax.set_title('Controller Performance Comparison', weight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_xlim(left=0, right=0.05)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='lightgrey')
    ax.grid(False)

    # --- Create a Zoomed-in Inset Plot ---
    ax_inset = ax.inset_axes([0.55, 0.08, 0.4, 0.4]) # [x, y, width, height]
    
    # Set the background color of the inset
    ax_inset.set_facecolor('whitesmoke')
    
    # Plot A2C and SAC data on the inset axes
    for algo in ['A2C', 'SAC']:
        if algo in df_rl.columns:
            ax_inset.plot(df_rl['Time (s)'], df_rl[algo], linewidth=1.2, color=rl_colors.get(algo))

    # Add error bounds to the inset plot
    ax_inset.axhline(y=upper_bound, color='k', linestyle='--', linewidth=1.5)
    ax_inset.axhline(y=lower_bound, color='k', linestyle='--', linewidth=1.5)
            
    # Set the zoomed-in limits to show the initial rise
    ax_inset.set_xlim(0, 0.0015)
    ax_inset.set_ylim(21, 32)
    
    # Optional: Add grid and finer ticks for the inset
    ax_inset.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_inset.tick_params(axis='both', which='major', labelsize=10)
    
    # Add a visible border to the inset
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    # --- Save the plot ---
    output_filename = 'controller_performance_comparison_with_inset.svg'
    plt.tight_layout()
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    print(f"\n📈 Plot saved successfully as '{output_filename}'")
    #plt.show()


if __name__ == '__main__':
    # Define paths for the two CSV files
    rl_controllers_csv = 'combined_evaluation_voltages.csv'
    pid_controller_csv = 'plot_data_noise_0.0_episode_1.csv'

    # Call the function with both file paths
    plot_combined_performance(rl_file_path=rl_controllers_csv, pid_file_path=pid_controller_csv)
