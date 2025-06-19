import os
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import TD3, A2C
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import the final, efficient custom environment class
from BCSimulinkEnv import BCSimulinkEnv

class EpisodeStatsLogger(BaseCallback):
    """
    A custom callback that logs episode statistics to the console and a text file.
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
    # --- Training Parameters ---
    total_timesteps = 25000      # Increased total training time
    LEARNING_STARTS = 600        # Random exploration for the first episode
    EPISODE_TIME = 0.03          # Corresponds to 600 agent steps
    GRACE_PERIOD = 50            # Short grace period for every episode startup

    print("Starting the environment...")

    # --- Environment Instantiation with Final Recommended Settings ---
    env_fn = lambda: BCSimulinkEnv(
        model_name="bcSim",
        frame_skip=10,
        enable_plotting=True,           # Disable plotting for fast training
        grace_period_steps=GRACE_PERIOD, # Use the short, fixed grace period
        max_episode_time=EPISODE_TIME,   # Use the longer episode time
        target_voltage=24.0
    )

    # Wrap the environment for use with Stable Baselines3
    env = DummyVecEnv([env_fn])
    env = VecNormalize(env,
                       norm_obs=True,      # Normalize observations
                       norm_reward=False,  # Do NOT normalize rewards
                       clip_obs=10.0)

    log_file_path = "a2c_bc_training_log.txt"

    print("Creating the A2C model...") 
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1
    )

    # --- Training ---
    custom_callback = EpisodeStatsLogger(log_path=log_file_path)
    print(f"--- Starting Training for {total_timesteps} Timesteps ---")
    print(f"Logging episode stats to: {log_file_path}")

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=custom_callback
    )

    # --- Saving ---
    model.save("a2c_bc_model_final")  # CHANGED NAME
    env.save("vec_normalize_stats_final.pkl")
    print("\n--- Model and Normalization Stats Saved ---")

    # Close the environment (and the MATLAB engine)
    env.close()


    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    # --- Model Definition ---
    # print("Creating the TD3 model...")
    # model = TD3(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=1e-4, #Was 1e-5
    #     buffer_size=250_000,
    #     batch_size=256,
    #     tau=0.005,
    #     gamma=0.99,
    #     train_freq=(1, "step"),
    #     learning_starts=LEARNING_STARTS,
    #     action_noise=action_noise,
    #     verbose=1,
    # )

    # # --- Training ---
    # custom_callback = EpisodeStatsLogger(log_path=log_file_path)
    # print(f"--- Starting Training for {total_timesteps} Timesteps ---")
    # print(f"Logging episode stats to: {log_file_path}")

    # model.learn(
    #     total_timesteps=total_timesteps,
    #     progress_bar=True,
    #     callback=custom_callback
    # )

    # # --- Saving ---
    # model.save("td3_bbc_model_final")
    # env.save("vec_normalize_stats_final.pkl")
    # print("\n--- Model and Normalization Stats Saved ---")

    # # Close the environment (and the MATLAB engine)
    # env.close()