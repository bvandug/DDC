import os
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC, A2C, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from BCSimulinkEnv import BCSimulinkEnv

class EpisodeStatsLogger(BaseCallback):
    """
    A custom callback that logs episode statistics and can append to a file.
    """
    def __init__(self, log_path: str, append: bool = False, verbose: int = 0):
        super(EpisodeStatsLogger, self).__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.file_mode = 'a' if append else 'w'
        self.episodes_so_far = 0
        
        # Add local trackers for reward and length
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_training_start(self) -> None:
        # If appending, count existing episodes to number correctly
        if self.file_mode == 'a' and os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    # Count lines that start with a digit (our episode lines)
                    self.episodes_so_far = sum(1 for line in f if line.strip() and line.strip()[0].isdigit())
            except Exception as e:
                print(f"[Warning] Could not read previous episode count from log: {e}")

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
        # Manually accumulate reward and length at each step. This is more robust.
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        if self.locals['dones'][0]:
            self.episodes_so_far += 1
            goal_voltage = self.training_env.get_attr('goal')[0]

            # Log to TensorBoard
            self.logger.record("rollout/ep_reward", self.current_episode_reward)

            # Log to console and file
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
    # --- CONFIGURATION ---
    # 1. Choose the model type you were training
    MODEL_TO_CONTINUE = 'SAC'
    
    # 2. Specify which checkpoint to load from
    CHECKPOINT_TIMESTEPS = 20000
    
    # 3. Specify how many MORE timesteps you want to train
    ADDITIONAL_TIMESTEPS = 80000
    
    # --- Set up paths ---
    # Path for SAVING logs and final models to Google Drive
    base_drive_path = f"/content/drive/MyDrive/DDC/{MODEL_TO_CONTINUE}_Randomized/"
    os.makedirs(base_drive_path, exist_ok=True) # Ensure save directory exists

    # Paths for LOADING the model and stats from Google Drive
    model_name_prefix = f"{MODEL_TO_CONTINUE.lower()}_bc_model_checkpoint"
    MODEL_PATH = os.path.join(base_drive_path, f"{model_name_prefix}_{CHECKPOINT_TIMESTEPS}_steps.zip")
    # CORRECTED the filename to match the format saved by the CheckpointCallback
    STATS_PATH = os.path.join(base_drive_path, f"{model_name_prefix}_vecnormalize_{CHECKPOINT_TIMESTEPS}_steps.pkl")
    
    # Paths for SAVING logs and future checkpoints to Google Drive
    LOG_PATH = os.path.join(base_drive_path, f"{MODEL_TO_CONTINUE.lower()}_bc_training_log.txt")
    TENSORBOARD_LOG_PATH = os.path.join(base_drive_path, f"{MODEL_TO_CONTINUE.lower()}_bc_tensorboard_log/")

    # --- Environment Instantiation ---
    print("--- Setting up environment to continue training ---")
    # THE FIX: Instantiate the environment with the randomized voltage range
    env_fn = lambda: BCSimulinkEnv(
        model_name="bcSim",
        enable_plotting=False,
        max_episode_time=0.1,
        target_voltage_min=25.0,
        target_voltage_max=35.0
    )

    print(f"Loading environment statistics from Google Drive path: {STATS_PATH}")
    env = DummyVecEnv([env_fn])
    env = VecNormalize.load(STATS_PATH, env)
    # IMPORTANT: Set the environment back to training mode
    env.training = True

    # --- Load the existing model ---
    print(f"Loading existing model from Google Drive path: {MODEL_PATH}")
    # Dynamically load the correct model class
    if MODEL_TO_CONTINUE == 'SAC':
        model = SAC.load(MODEL_PATH, env=env, tensorboard_log=TENSORBOARD_LOG_PATH)
    elif MODEL_TO_CONTINUE == 'TD3':
        model = TD3.load(MODEL_PATH, env=env, tensorboard_log=TENSORBOARD_LOG_PATH)
    elif MODEL_TO_CONTINUE == 'A2C':
        model = A2C.load(MODEL_PATH, env=env, tensorboard_log=TENSORBOARD_LOG_PATH)
    else:
        raise ValueError(f"Model type '{MODEL_TO_CONTINUE}' not recognized.")
        
    print(f"Model loaded. Current timesteps: {model.num_timesteps}")

    # --- Setup Callbacks for Continuation ---
    continue_log_callback = EpisodeStatsLogger(log_path=LOG_PATH, append=True)
    # Checkpoint callback will save to the Google Drive path
    checkpoint_callback = CheckpointCallback(
      save_freq=20000,
      save_path=base_drive_path,
      name_prefix=model_name_prefix,
      save_replay_buffer=True,
      save_vecnormalize=True,
    )

    # --- Continue Training ---
    print(f"--- Continuing Training for an additional {ADDITIONAL_TIMESTEPS} Timesteps ---")
    
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        progress_bar=True,
        callback=[continue_log_callback, checkpoint_callback],
        reset_num_timesteps=False  # IMPORTANT: Do not reset the trained steps counter
    )

    # --- Saving the final updated model and stats to Google Drive ---
    final_model_path = os.path.join(base_drive_path, f"{MODEL_TO_CONTINUE.lower()}_bc_model_final.zip")
    final_stats_path = os.path.join(base_drive_path, f"{MODEL_TO_CONTINUE.lower()}_vec_normalize_final.pkl")
    
    print(f"\n--- Saving Final Model to {final_model_path} ---")
    model.save(final_model_path)
    env.save(final_stats_path)
    print("Save complete.")

    # Close the environment
    env.close()
