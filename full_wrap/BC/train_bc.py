import os
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC, A2C, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer

# Import your custom environment class
# Ensure BCSimulinkEnv.py is in the same directory or accessible in the Python path.
from BCSimulinkEnv import BCSimulinkEnv

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
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_training_start(self) -> None:
        self.log_file = open(self.log_path, "w")
        header = f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "-"*70 + "\n"
        header += f"{'Episode':<10}{'Total Reward':<20}{'Episode Length':<20}{'Goal Voltage':<20}\n"
        header += "-"*70 + "\n"
        self.log_file.write(header); self.log_file.flush()
        print(header, end='')

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log to TensorBoard
            self.logger.record("rollout/ep_reward", self.current_episode_reward)
            
            # Log to console and file
            episode_num = len(self.episode_rewards)
            goal_voltage = self.training_env.get_attr('goal')[0]
            log_line = (f"{episode_num:<10}"
                        f"{self.current_episode_reward:<20.4f}"
                        f"{self.current_episode_length:<20}"
                        f"{goal_voltage:<20.4f}\n")
            print(log_line, end='')
            self.log_file.write(log_line); self.log_file.flush()

            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        return True

    def _on_training_end(self) -> None:
        footer = "-"*70 + "\n"
        footer += f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(footer, end='')
        self.log_file.write(footer)
        self.log_file.close()

if __name__ == "__main__":
    # --- CHOOSE THE MODEL TO TRAIN ---
    # Options: 'SAC', 'A2C', 'TD3'
    MODEL_TO_TRAIN = 'SAC'
    
    # --- Training Parameters ---
    total_timesteps = 100000
    EPISODE_TIME = 0.1
    GRACE_PERIOD = 50

    # --- Define Base Save Path in Google Drive ---
    base_drive_path = f"/content/drive/MyDrive/DDC/{MODEL_TO_TRAIN}/"
    os.makedirs(base_drive_path, exist_ok=True)

    log_file_path = os.path.join(base_drive_path, f"{MODEL_TO_TRAIN.lower()}_bc_training_log.txt")
    tensorboard_log_path = os.path.join(base_drive_path, f"{MODEL_TO_TRAIN.lower()}_bc_tensorboard_log/")

    print("Starting the environment...")
    # --- Environment Instantiation with a FIXED target voltage ---
    env_fn = lambda: BCSimulinkEnv(
        model_name="bcSim", 
        frame_skip=10, 
        enable_plotting=False,
        grace_period_steps=GRACE_PERIOD, 
        max_episode_time=EPISODE_TIME,
        target_voltage=30.0  # Training for a fixed target
    )
    env = DummyVecEnv([env_fn])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = None
    
    # --- MODEL CONFIGURATION ---
    if MODEL_TO_TRAIN == 'SAC':
        print("--- Configuring SAC Model ---")
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            learning_starts=10000,
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
            verbose=1, tensorboard_log=tensorboard_log_path
        )

    elif MODEL_TO_TRAIN == 'A2C':
        print("--- Configuring A2C Model ---")
        model = A2C(
            "MlpPolicy", env,
            learning_rate=7e-4,
            n_steps=20,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
            verbose=1, tensorboard_log=tensorboard_log_path
        )

    elif MODEL_TO_TRAIN == 'TD3':
        print("--- Configuring TD3 Model (Tuned for Stability) ---")
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
        
        model = TD3(
            "MlpPolicy", env,
            learning_rate=7.5e-5,
            buffer_size=1_000_000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            action_noise=action_noise,
            policy_delay=3,
            policy_kwargs=dict(net_arch=[400, 300]),
            verbose=1, tensorboard_log=tensorboard_log_path
        )

    else:
        raise ValueError(f"Model type '{MODEL_TO_TRAIN}' not recognized. Choose from 'SAC', 'A2C', 'TD3'.")

    # --- Setup Callbacks ---
    custom_callback = EpisodeStatsLogger(log_path=log_file_path)
    checkpoint_callback = CheckpointCallback(
      save_freq=20000,
      save_path=base_drive_path,
      name_prefix=f"{MODEL_TO_TRAIN.lower()}_bc_model_checkpoint",
      save_replay_buffer=True,
      save_vecnormalize=True,
    )

    # --- Training ---
    print(f"--- Starting Training for {MODEL_TO_TRAIN} ({total_timesteps} Timesteps) ---")
    print(f"Logging episode stats to: {log_file_path}")
    print(f"Saving model checkpoints to: {base_drive_path}")
    print(f"Saving TensorBoard logs to: {tensorboard_log_path}")

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=[custom_callback, checkpoint_callback]
    )

    # --- Final Saving ---
    model.save(os.path.join(base_drive_path, f"{MODEL_TO_TRAIN.lower()}_bc_model_final"))
    env.save(os.path.join(base_drive_path, f"{MODEL_TO_TRAIN.lower()}_vec_normalize_final.pkl"))
    print(f"\n--- Final {MODEL_TO_TRAIN} Model and Stats Saved to Google Drive ---")

    env.close()
