import pandas as pd
import matplotlib.pyplot as plt

# --- Global Font and Style Configuration ---
# We've removed the tick settings from here to apply them directly later.
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'axes.titleweight': 'bold',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',       # This sets the tick LABEL color
    'ytick.color': 'black',       # This sets the tick LABEL color
    'axes.edgecolor': 'black',
    'legend.labelcolor': 'black',
})

def plot_controller_performance(file_path, goal_voltage=30.0, error_margin=0.5):
    """
    Reads controller performance data from an XLSX file and generates a plot
    similar to the provided example. It automatically converts a time column
    from milliseconds to seconds.

    Args:
        file_path (str): The path to the XLSX data file.
        goal_voltage (float): The target voltage for the controller.
        error_margin (float): The acceptable voltage error margin.
    """
    try:
        df = pd.read_excel(file_path, sheet_name='evaluation_results_30.0_A2C_dat')
        print("Successfully loaded the Excel file.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return

    # --- Time Conversion ---
    time_col_ms = 'time_ms'
    time_col_s = 'Time (s)'

    if time_col_ms not in df.columns:
        print(f"Error: Time column '{time_col_ms}' not found in the Excel file.")
        time_col_ms = 'Time (ms)'
        if time_col_ms not in df.columns:
            print(f"Error: Fallback time column '{time_col_ms}' also not found.")
            return

    df[time_col_s] = df[time_col_ms] / 1000.0
    print(f"Converted time column '{time_col_ms}' from milliseconds to seconds.")

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Manually set the outline (spines) to black ---
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    
    # --- NEW: Force tick marks to be visible ---
    # This overrides the style sheet and ensures ticks are drawn.
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.2)


    # --- Plot Algorithm Data ---
    algorithms = ['A2C', 'SAC']
    for algo in algorithms:
        if algo in df.columns:
            ax.plot(df[time_col_s], df[algo], label=f'{algo} Performance', linewidth=1.5)
        else:
            print(f"Warning: Column for algorithm '{algo}' not found. It will be skipped.")

    # --- Plot Goal and Error Bounds ---
    upper_bound = goal_voltage + error_margin
    lower_bound = goal_voltage - error_margin

    ax.axhline(y=upper_bound, color='k', linestyle='--', linewidth=1.5, label=f'±{error_margin}V Error Bound')
    ax.axhline(y=lower_bound, color='k', linestyle='--', linewidth=1.5)

    # --- Final Touches and Labels ---
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_xlim(left=0, right=0.003)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, facecolor='white', edgecolor='lightgrey')
    ax.grid(False)

    # --- Save the plot ---
    output_filename = 'controller_performance_BC.svg'
    plt.tight_layout()
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    print(f"Plot saved successfully as {output_filename}")


if __name__ == '__main__':
    excel_file_path = 'eval_A2C_SAC.xlsx'
    plot_controller_performance(file_path=excel_file_path)