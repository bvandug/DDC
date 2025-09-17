import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def plot_combined_performance(rl_file_path, pid_file_path, goal_voltage=-30.0, error_percentage=4.0):
    """
    Reads performance data from two XLSX files and generates a single,
    comparative plot with a zoomed-in inset.
    """
    # --- Load Data from RL Controllers Excel File ---
    try:
        # ** FIX: Added `decimal=','` to handle comma-separated decimals **
        df_rl = pd.read_excel(rl_file_path, sheet_name=0, decimal=',')
        print(f"✅ Successfully loaded the RL data file: '{rl_file_path}'")
        # Rename columns by position to ensure consistency
        df_rl.columns = ['Time (s)', 'SAC', 'A2C']
    except FileNotFoundError:
        print(f"❌ Error: The file '{rl_file_path}' was not found.")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the RL Excel file: {e}")
        return

    # --- Load Data from PID Controller Excel File ---
    try:
        # ** FIX: Added `decimal=','` to handle comma-separated decimals **
        df_pid = pd.read_excel(pid_file_path, sheet_name=0, decimal=',')
        print(f"✅ Successfully loaded the PID data file: '{pid_file_path}'")
    except FileNotFoundError:
        print(f"❌ Error: The PID file '{pid_file_path}' was not found.")
        return
    except Exception as e:
        print(f"❌ An error occurred while reading the PID Excel file: {e}")
        return

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
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
    if 'Time (s)' in df_pid.columns and 'Voltage' in df_pid.columns:
        ax.plot(df_pid['Time (s)'], df_pid['Voltage'], label='PID Controller', linewidth=1.2, color='green')
    else:
        print("⚠️ Warning: PID XLSX must contain 'Time (s)' and 'Voltage' columns. Skipping PID plot.")

    # --- Plot Goal and Error Bounds ---
    error_margin = goal_voltage * (error_percentage / 100.0)
    upper_bound = goal_voltage + error_margin
    lower_bound = goal_voltage - error_margin
    ax.axhline(y=upper_bound, color='k', linestyle='--', linewidth=1.5, label=f'±{error_percentage}% Error Bound')
    ax.axhline(y=lower_bound, color='k', linestyle='--', linewidth=1.5)

    # --- Final Touches and Labels for Main Plot ---
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_xlim(0, 0.05)
    ax.set_ylim(-80, 10)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='lightgrey')
    ax.grid(False)

    # --- Create a Zoomed-in Inset Plot ---
    # After (Top-Right)
    ax_inset = ax.inset_axes([0.6, 0.63, 0.35, 0.35])
    ax_inset.set_facecolor('whitesmoke')
    
    for algo in ['A2C', 'SAC']:
        if algo in df_rl.columns:
            ax_inset.plot(df_rl['Time (s)'], df_rl[algo], linewidth=1.2, color=rl_colors.get(algo))

    ax_inset.axhline(y=upper_bound, color='k', linestyle='--', linewidth=1.5)
    ax_inset.axhline(y=lower_bound, color='k', linestyle='--', linewidth=1.5)
            
    #ax_inset.set_xscale('log')
    ax_inset.set_xlim(1e-4, 0.0020)
    ax_inset.set_ylim(goal_voltage - (error_margin * 5), goal_voltage + (error_margin * 1.5))
    # Add this line to invert the y-axis
    ax_inset.invert_yaxis()
    ax_inset.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_inset.tick_params(axis='both', which='major', labelsize=10)
    
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
    

    # --- Save the plot ---
    output_filename = 'controller_performance_final_BBC.svg'
    plt.tight_layout()
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    print(f"\n📈 Plot saved successfully as '{output_filename}'")

if __name__ == '__main__':
    rl_controllers_xlsx = 'BBCData_SAC_A2C.xlsx'
    pid_controller_xlsx = 'BBCData_PID.xlsx'
    plot_combined_performance(rl_file_path=rl_controllers_xlsx, pid_file_path=pid_controller_xlsx)