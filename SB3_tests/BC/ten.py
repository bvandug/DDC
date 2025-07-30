import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt

def extract_and_plot_tensorboard_data(log_file_path, scalar_tag='rollout/ep_rew_mean'):
    """
    Extracts all data points for a specific scalar from a TensorBoard tfevents file
    and then plots the results.

    :param log_file_path: The full path to the tfevents file.
    :param scalar_tag: The name of the scalar to extract (e.g., 'rollout/ep_rew_mean').
    """
    print(f"--- Loading data from: {log_file_path} ---")
    
    # Check if the file exists before proceeding
    if not os.path.exists(log_file_path):
        print(f"Error: The file was not found at the specified path: {log_file_path}")
        print("Please make sure the path is correct.")
        return

    # Initialize the EventAccumulator to load all data
    event_acc = EventAccumulator(log_file_path, size_guidance={'scalars': 0})
    event_acc.Reload()
    
    # Check if the requested tag is available
    if scalar_tag not in event_acc.Tags()['scalars']:
        print(f"Error: Tag '{scalar_tag}' not found in the log file.")
        print("Available scalar tags are:", event_acc.Tags()['scalars'])
        return

    # Extract the scalar events (wall_time, step, value)
    scalar_events = event_acc.Scalars(scalar_tag)
    
    # Create lists for plotting
    steps = [event.step for event in scalar_events]
    rewards = [event.value for event in scalar_events]
    
    print(f"\n--- Extracted {len(steps)} data points for tag '{scalar_tag}' ---")

    # --- Plotting the Data ---
    print("--- Generating plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(steps, rewards, marker='o', linestyle='-', color='b', label='Mean Episode Reward')
    
    ax.set_title('Training Progress: Mean Episode Reward vs. Timesteps', fontsize=16, weight='bold')
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Mean Episode Reward (ep_rew_mean)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()
    
    # --- Optional: Save data to CSV ---
    df = pd.DataFrame({'step': steps, 'reward': rewards})
    csv_path = os.path.splitext(log_file_path)[0] + '_full_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n--- Full data also saved to: {csv_path} ---")


if __name__ == '__main__':
    # --- IMPORTANT: UPDATE THIS PATH ---
    # Replace this placeholder with the full path to your .events file
    # It will be inside a folder like 'DQN_1', 'PPO_1', etc.
    log_file_path = '/Users/nicholas/Documents/MATLAB/MATLAB/full_wrap/BC/tensorboard_logs/A2C_1/events.out.tfevents.1752670380.28782a754a70.53739.0'
    
    if 'YOUR_FULL_PATH' in log_file_path:
        print("Please update the 'log_file_path' variable with the full path to your event file.")
    else:
        extract_and_plot_tensorboard_data(log_file_path)
