import os
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import your custom environment class
from BCSimulinkEnv import BCSimulinkEnv

# --- ROBUST EpisodeStatsLogger for Continuing Training ---
class EpisodeStatsLogger(BaseCallback):
    """A robust custom callback that logs episode statistics and can append to a file."""
    def __init__(self, log_path: str, append: bool = False, verbose: int = 0):
        super(EpisodeStatsLogger, self).__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.file_mode = 'a' if append else 'w'
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.episodes_so_far = 0

    def _on_training_start(self) -> None:
        # If appending, count existing episodes to number correctly
        if self.file_mode == 'a' and os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    # Heuristic to count lines that are episode summaries
                    self.episodes_so_far = sum(1 for line in f if line.strip() and line.strip()[0].isdigit())
            except Exception as e:
                print(f"Could not read previous episode count from log: {e}")

        self.log_file = open(self.log_path, self.file_mode)
        
        if self.file_mode == 'w':
            header = f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            header += "-"*70 + "\n"
            header += f"{'Episode':<10}{'Total Reward':<20}{'Episode Length':<20}{'Goal Voltage':<20}\n"
            header += "-"*70 + "\n"
            self.log_file.write(header); self.log_file.flush()
            print(header, end='')
        else:
            append_header = f"\n--- Continuation training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            self.log_file.write(append_header); self.log_file.flush()
            print(f"Appending logs to {self.log_path}. Starting from Episode {self.episodes_so_far + 1}.")

    def _on_step(self) -> bool:
        # Manually accumulate reward at each step. This is the robust method.
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        # Check if the episode is done
        if self.locals['dones'][0]:
            self.episodes_so_far += 1
            goal_voltage = self.training_env.get_attr('goal')[0]

            log_line = (f"{self.episodes_so_far:<10}"
                        f"{self.current_episode_reward:<20.4f}"
                        f"{self.current_episode_length:<20}"
                        f"{goal_voltage:<20.4f}\n")

            print(log_line, end='')
            self.log_file.write(log_line); self.log_file.flush()

            # Reset for the next episode
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        return True

    def _on_training_end(self) -> None:
        footer = "-"*70 + "\n"
        footer += f"Continuation training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(footer, end='')
        if self.log_file:
            self.log_file.write(footer)
            self.log_file.close()


if __name__ == "__main__":
    # --- File Paths ---
    MODEL_PATH = "a2c_bc_model_final.zip"
    STATS_PATH = "vec_normalize_stats_final.pkl"
    LOG_PATH = "a2c_bc_training_log.txt"
    
    # --- How many MORE steps to train ---
    additional_timesteps = 25000

    # --- Environment Instantiation ---
    # Use a lambda to pass arguments to the environment constructor
    env_fn = lambda: BCSimulinkEnv(
        model_name="bcSim",
        enable_plotting=True # Disable plotting for faster training
    )

    print(f"Loading environment statistics from: {STATS_PATH}")
    env = DummyVecEnv([env_fn])
    env = VecNormalize.load(STATS_PATH, env)
    # IMPORTANT: Set the environment back to training mode
    env.training = True

    # --- Load the existing model ---
    print(f"Loading existing model from: {MODEL_PATH}")
    model = A2C.load(MODEL_PATH, env=env)

    # --- Continue Training ---
    # Use the corrected logger in append mode
    custom_callback = EpisodeStatsLogger(log_path=LOG_PATH, append=True)
    print(f"--- Continuing Training for an additional {additional_timesteps} Timesteps ---")
    
    model.learn(
        total_timesteps=additional_timesteps,
        progress_bar=True,
        callback=custom_callback,
        reset_num_timesteps=False  # IMPORTANT: Do not reset the trained steps counter
    )

    # --- Saving the updated model and stats ---
    print("\n--- Saving Updated Model and Normalization Stats ---")
    model.save(MODEL_PATH)
    env.save(STATS_PATH)
    print("Save complete.")

    # Close the environment
    env.close()