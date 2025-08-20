import pandas as pd
import matplotlib.pyplot as plt

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

def plot_controller_performance(file_path, goal_voltage=30.0, error_margin=0.5):
    """
    Reads controller performance data from an XLSX file and generates a plot.
    Expects time to already be in seconds.

    Args:
        file_path (str): The path to the XLSX data file.
        goal_voltage (float): The target voltage for the controller.
        error_margin (float): The acceptable voltage error margin.
    """
    try:
        df = pd.read_excel(file_path, sheet_name='evaluation_on_env_noise_0_01_da')
        print("Successfully loaded the Excel file.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")
        return

    time_col = 'Time (s)'  # Expecting this column already in seconds

    if time_col not in df.columns:
        print(f"Error: Expected time column '{time_col}' not found in the Excel file.")
        return

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.2)

    # --- Plot Algorithm Data ---
    algorithms = ['A2C', 'SAC']
    for algo in algorithms:
        if algo in df.columns:
            ax.plot(df[time_col], df[algo], label=f'{algo} Performance', linewidth=1.5)
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
    excel_file_path = 'eval_data_noise.xlsx'
    plot_controller_performance(file_path=excel_file_path)
