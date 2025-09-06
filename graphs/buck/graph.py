import json
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the JSON files
with open('DQN_DQN_Rank_1_Seed_42_Noise_0_tensorboard_logs_DQN_1.json', 'r') as f:
    dqn_data = json.load(f)

with open('TD3_TD3_Rank_1_Seed_42_Noise_0_tensorboard_logs_TD3_1.json', 'r') as f:
    td3_data = json.load(f)

# Create pandas DataFrames
dqn_df = pd.DataFrame(dqn_data, columns=['Time', 'Timesteps', 'Reward'])
td3_df = pd.DataFrame(td3_data, columns=['Time', 'Timesteps', 'Reward'])

# --- PLOTTING ---
plt.figure(figsize=(12, 6))

# Set the global font size for the plot
plt.rcParams.update({'font.size': 18}) 

# Plot data
plt.plot(dqn_df['Timesteps'], dqn_df['Reward'], label='DQN', color='blue')
plt.plot(td3_df['Timesteps'], td3_df['Reward'], label='TD3', color='red')

# Add enhancements
plt.title('Buck Converter DQN vs. TD3 Training Convergence')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('BC_dqn_vs_td3_performance.pdf', bbox_inches='tight')

print("Plot with larger text saved as BC_dqn_vs_td3_performance.pdf")