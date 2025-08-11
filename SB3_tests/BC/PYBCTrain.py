import os
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC, A2C, TD3, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

from PYBCEnv import BuckConverterEnv

class EpisodeStatsLogger(BaseCallback):
    """
    A custom callback that logs episode statistics to the console, a text file,
    and to TensorBoard.
    """
    def __init__(self, log_path: str, verbose: int = 0):
        super(EpisodeStatsLogger, self).__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_rewards = None
        self.current_episode_lengths = None

    def _on_training_start(self) -> None:
        self.log_file = open(self.log_path, "w")
        header = f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "-"*80 + "\n"
        header += f"{'Episode':<10}{'Total Reward':<20}{'Episode Length':<20}{'Goal Voltage':<20}\n"
        header += "-"*80 + "\n"
        self.log_file.write(header)
        self.log_file.flush()
        print(header, end='')
        self.current_episode_rewards = np.zeros(self.training_env.num_envs)
        self.current_episode_lengths = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        self.current_episode_rewards += self.locals['rewards']
        self.current_episode_lengths += 1
        
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_lengths.append(self.current_episode_lengths[i])
                
                self.logger.record("rollout/ep_reward", self.current_episode_rewards[i])
                self.logger.record("rollout/ep_length", self.current_episode_lengths[i])
                
                episode_num = len(self.episode_rewards)
                goal_voltage = self.training_env.get_attr('goal', indices=[i])[0]
                
                log_line = (f"{episode_num:<10}"
                            f"{self.current_episode_rewards[i]:<20.4f}"
                            f"{int(self.current_episode_lengths[i]):<20}"
                            f"{goal_voltage:<20.4f}\n")
                print(log_line, end='')
                self.log_file.write(log_line)
                self.log_file.flush()

                self.current_episode_rewards[i] = 0
                self.current_episode_lengths[i] = 0
        return True

    def _on_training_end(self) -> None:
        footer = "-"*80 + "\n"
        footer += f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(footer, end='')
        self.log_file.write(footer)
        self.log_file.close()

if __name__ == "__main__":
    if IS_COLAB:
        print("Running in Google Colab. Assuming Drive is already mounted.")
        # Define the base path inside Google Drive
        drive_base_path = "/content/drive/MyDrive/DDC"
    else:
        # If not in Colab, save to a local folder named 'DDC'
        drive_base_path = "./DDC_Temp"

    # CHOOSE THE MODEL TO TRAIN
    MODEL_TO_TRAIN = 'A2C'
    USE_RANDOMIZED_VOLTAGE = True
    FIXED_VOLTAGE = 30.0
    total_timesteps = 400000 
    MAX_EPISODE_STEPS = 600
    GRACE_PERIOD = 50

    training_mode = "Randomized" if USE_RANDOMIZED_VOLTAGE else "Fixed"
    base_save_path = os.path.join(drive_base_path, f"{MODEL_TO_TRAIN}_{training_mode}_Python_1")
    os.makedirs(base_save_path, exist_ok=True)
    print(f"All files will be saved to: {base_save_path}")

    log_file_path = os.path.join(base_save_path, f"{MODEL_TO_TRAIN.lower()}_bc_training_log.txt")
    tensorboard_log_path = os.path.join(base_save_path, f"{MODEL_TO_TRAIN.lower()}_bc_tensorboard_log/")

    print("Configuring the environment...")
    if USE_RANDOMIZED_VOLTAGE:
        print("--- Mode: Randomized Voltage (20V-35V) ---")
        env_fn = lambda: BuckConverterEnv(
            frame_skip=10, grace_period_steps=GRACE_PERIOD, max_episode_steps=MAX_EPISODE_STEPS,
            use_randomized_goal=True, target_voltage_min=28.5, target_voltage_max=31.5, voltage_noise_std = 0.01
        )
    else:
        print(f"--- Mode: Fixed Voltage ({FIXED_VOLTAGE}V) ---")
        env_fn = lambda: BuckConverterEnv(
            frame_skip=10, grace_period_steps=GRACE_PERIOD, max_episode_steps=MAX_EPISODE_STEPS,
            use_randomized_goal=False, fixed_goal_voltage=FIXED_VOLTAGE
        )

    env = DummyVecEnv([env_fn])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model_class = globals()[MODEL_TO_TRAIN]
    model = model_class("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_path)

    custom_callback = EpisodeStatsLogger(log_path=log_file_path)
    checkpoint_callback = CheckpointCallback(
      save_freq=20000, save_path=base_save_path,
      name_prefix=f"{MODEL_TO_TRAIN.lower()}_bc_model_checkpoint",
      save_replay_buffer=True, save_vecnormalize=True,
    )

    print(f"\n--- Starting Training for {MODEL_TO_TRAIN} ({total_timesteps} Timesteps) ---")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=[custom_callback, checkpoint_callback]
    )

    final_model_path = os.path.join(base_save_path, f"{MODEL_TO_TRAIN.lower()}_bc_model_final")
    final_env_stats_path = os.path.join(base_save_path, f"{MODEL_TO_TRAIN.lower()}_vec_normalize_final.pkl")
    model.save(final_model_path)
    env.save(final_env_stats_path)
    print(f"\n--- Final {MODEL_TO_TRAIN} Model and Stats Saved to '{base_save_path}' ---")
    
    # Close the training environment
    env.close()

    # ===================================================================
    # --- EVALUATION SECTION ---
    # ===================================================================
    print("\n--- Starting Evaluation of the Trained Model ---")
    
    # Load the trained agent and normalization stats
    eval_model = model_class.load(final_model_path)
    eval_env_stats = VecNormalize.load(final_env_stats_path, DummyVecEnv([env_fn]))
    
    # Set the loaded environment to evaluation mode
    eval_env_stats.training = False
    eval_env_stats.norm_reward = False

    # Create a new environment instance for evaluation WITHOUT rendering
    eval_env = BuckConverterEnv(
        render_mode=None, # Set to None to prevent automatic plotting
        use_randomized_goal=False, # Use a fixed goal for a clear test
        fixed_goal_voltage=28.0 # Test a specific voltage
    )
    
    obs, info = eval_env.reset()
    print(f"Evaluating with a fixed target of {eval_env.goal:.2f}V")

    # Data collection lists for manual plotting
    eval_times = []
    eval_voltages = []
    eval_duties = []
    eval_goals = []

    try:
        for _ in range(MAX_EPISODE_STEPS):
            # Important: The observation from the raw environment must be normalized
            normalized_obs = eval_env_stats.normalize_obs(obs)
            
            action, _states = eval_model.predict(normalized_obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Collect data for plotting
            eval_times.append(eval_env.total_sim_time)
            eval_voltages.append(obs[0]) # Voltage is the first element of obs
            eval_duties.append(action[0])
            eval_goals.append(eval_env.goal)

            if terminated or truncated:
                print("Evaluation episode finished.")
                break
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        # --- Manual Plotting and Saving Section ---
        print("\nPlotting and saving evaluation results...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"Evaluation Result - Target: {eval_env.goal:.2f}V", fontsize=16)

        ax1.plot(eval_times, eval_voltages, 'b-', label='Actual Voltage')
        ax1.plot(eval_times, eval_goals, 'r--', label='Target Voltage')
        ax1.set_ylabel("Voltage (V)")
        ax1.grid(True)
        ax1.legend()

        ax2.plot(eval_times, eval_duties, 'm-', label='Duty Cycle (Action)')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Duty Cycle")
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        image_save_path = os.path.join(base_save_path, "evaluation_plot.png")
        print(f"Saving final evaluation plot to: {image_save_path}")
        fig.savefig(image_save_path)
        print("Plot saved successfully.")
        
        # Show the plot in an interactive window
        plt.show()

        eval_env.close()
