import os
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Import the custom Simulink environment class
# Make sure 'BBCSimulink_env.py' is in the same directory
from BBCSimulink_env import BBCSimulinkEnv

# --- Definition of the Custom Callback for Episode Logging ---
class EpisodeStatsLogger(BaseCallback):
    """
    A custom callback that logs episode statistics (reward, length) to the console
    and to a text file at the end of each episode.
    """
    def __init__(self, log_path: str, verbose: int = 0):
        super(EpisodeStatsLogger, self).__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        We use it to open the log file and write the header.
        """
        self.log_file = open(self.log_path, "w")
        header = f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "-"*70 + "\n"
        header += f"{'Episode':<10}{'Total Reward':<20}{'Episode Length':<20}{'Goal Voltage':<20}\n"
        header += "-"*70 + "\n"
        self.log_file.write(header)
        self.log_file.flush()
        print(header, end='')

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        """
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        self.current_episode_length += 1

        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            episode_num = len(self.episode_rewards)
            goal_voltage = self.training_env.get_attr('goal')[0]

            log_line = (f"{episode_num:<10}"
                        f"{self.current_episode_reward:<20.4f}"
                        f"{self.current_episode_length:<20}"
                        f"{goal_voltage:<20.4f}\n")

            print(log_line, end='')
            self.log_file.write(log_line)
            self.log_file.flush()

            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        return True

    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        """
        footer = "-"*70 + "\n"
        footer += f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(footer, end='')
        self.log_file.write(footer)
        self.log_file.close()


# --- Main execution block ---
if __name__ == "__main__":
    # 1. Instantiate the Simulink Environment
    print("Starting the environment...")
    env = make_vec_env(BBCSimulinkEnv, n_envs=1)

    # 2. Define training parameters
    # Set timesteps to 27,000 to run for exactly 9 episodes (3000 steps/episode)
    total_timesteps = 27000
    log_file_path = "td3_bbc_training_log.txt"

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    print("Creating the model...")
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(10, "step"), #Train every 10 steps
        action_noise=action_noise,
        verbose=1,
    )

    # 4. Instantiate the custom callback for logging
    custom_callback = EpisodeStatsLogger(log_path=log_file_path)

    # 5. Train the agent
    print(f"--- Starting Training for {total_timesteps} Timesteps ---")
    print(f"Logging episode stats to: {log_file_path}")

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=custom_callback
    )

    # 6. Save the trained model
    model.save("td3_bbc_model_tuned")
    print("\n--- Model Saved as td3_bbc_model_tuned.zip ---")

    # 7. Close the environment (and the MATLAB engine)
    env.close()