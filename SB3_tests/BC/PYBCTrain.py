import os
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC, A2C, TD3, PPO, DDPG, DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from PYBCEnv import BuckConverterEnv, DiscretizeActionWrapper


class EpisodeStatsLogger(BaseCallback):
    """A custom callback to log episode statistics during training.

    This callback records the reward, length, and goal voltage for each
    episode, printing them to the console and saving them to a log file.
    It also integrates with the Stable Baselines3 logger to record metrics
    for TensorBoard.

    Attributes:
        log_path (str): Path to the output text log file.
    """
    def __init__(self, log_path: str, verbose: int = 0):
        """Initializes the EpisodeStatsLogger callback.

        Args:
            log_path (str): The file path where the training log will be saved.
            verbose (int): The verbosity level (0 or 1).
        """
        super().__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_rewards = None
        self.current_episode_lengths = None

    def _on_training_start(self) -> None:
        """Called once at the beginning of training."""
        self.log_file = open(self.log_path, "w")
        header = f"Training started at: " \
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "-" * 80 + "\n"
        header += (f"{'Episode':<10}{'Total Reward':<20}"
                   f"{'Episode Length':<20}{'Goal Voltage':<20}\n")
        header += "-" * 80 + "\n"
        self.log_file.write(header)
        self.log_file.flush()
        print(header, end='')
        self.current_episode_rewards = np.zeros(self.training_env.num_envs)
        self.current_episode_lengths = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        """Called after each step in the environment.

        Checks for completed episodes and logs their stats.

        Returns:
            bool: True to continue training, False to stop.
        """
        self.current_episode_rewards += self.locals['rewards']
        self.current_episode_lengths += 1

        for i, done in enumerate(self.locals['dones']):
            if done:
                reward = self.current_episode_rewards[i]
                length = self.current_episode_lengths[i]
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)

                # Log to TensorBoard
                self.logger.record("rollout/ep_reward", reward)
                self.logger.record("rollout/ep_length", length)

                episode_num = len(self.episode_rewards)
                goal_voltage = self.training_env.get_attr('goal',
                                                          indices=[i])[0]

                # Log to console and file
                log_line = (f"{episode_num:<10}"
                            f"{reward:<20.4f}"
                            f"{int(length):<20}"
                            f"{goal_voltage:<20.4f}\n")
                print(log_line, end='')
                self.log_file.write(log_line)
                self.log_file.flush()

                # Reset counters for the next episode
                self.current_episode_rewards[i] = 0
                self.current_episode_lengths[i] = 0
        return True

    def _on_training_end(self) -> None:
        """Called once at the end of training."""
        footer = "-" * 80 + "\n"
        footer += f"Training finished at: " \
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        print(footer, end='')
        self.log_file.write(footer)
        self.log_file.close()


if __name__ == "__main__":
    MODELS_TO_TRAIN = ['A2C', 'PPO', 'DQN', 'SAC', 'TD3', 'DDPG']
    NOISE_LEVELS = [0, 0.001, 0.01, 0.1]
    TOTAL_TIMESTEPS = 400000
    MAX_EPISODE_STEPS = 600
    GRACE_PERIOD = 50
    SEED = 42

    # Path Configuration for Local Execution
    BASE_SAVE_PATH = "models"
    RESULTS_LOG_DIR = "PY_BC_Results"

    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_LOG_DIR, exist_ok=True)

    # Main Training Loop
    for model_type in MODELS_TO_TRAIN:
        for noise in NOISE_LEVELS:
            print("\n" + "=" * 80)
            print(f"--- STARTING TRAINING: {model_type} | Noise: {noise} ---")
            print("=" * 80 + "\n")

            # Define unique save paths for this training run
            save_folder_name = f"{model_type}_Seed_{SEED}_Noise_{noise}"
            model_save_path = os.path.join(BASE_SAVE_PATH, model_type,
                                           save_folder_name)
            log_save_path = os.path.join(RESULTS_LOG_DIR, model_type)
            os.makedirs(model_save_path, exist_ok=True)
            os.makedirs(log_save_path, exist_ok=True)

            log_file_path = os.path.join(
                log_save_path, f"train_log_{model_type}_noise_{noise}.txt"
            )
            tensorboard_log_path = os.path.join(log_save_path, "tensorboard/")

            env = None
            try:
                # Environment Setup
                def create_env_fn(current_noise):
                    def _init():
                        return BuckConverterEnv(
                            frame_skip=10,
                            grace_period_steps=GRACE_PERIOD,
                            max_episode_steps=MAX_EPISODE_STEPS,
                            use_randomized_goal=True,
                            target_voltage_min=28.5,
                            target_voltage_max=31.5,
                            voltage_noise_std=current_noise
                        )
                    return _init

                base_env_fn = create_env_fn(noise)

                if model_type == 'DQN':
                    env_fn = lambda: DiscretizeActionWrapper(base_env_fn(),
                                                             n_bins=17)
                else:
                    env_fn = base_env_fn

                env = DummyVecEnv([env_fn])
                env = VecNormalize(env, norm_obs=True, norm_reward=False,
                                   clip_obs=10.0)

                # Model Setup
                model_class_map = {
                    'SAC': SAC, 'TD3': TD3, 'A2C': A2C,
                    'DDPG': DDPG, 'PPO': PPO, 'DQN': DQN
                }
                model_class = model_class_map.get(model_type)
                if model_class is None:
                    raise ValueError(f"Model type '{model_type}' not recognized.")

                model = model_class("MlpPolicy", env, verbose=0, seed=SEED,
                                    tensorboard_log=tensorboard_log_path)

                # Custom Callbacks
                custom_callback = EpisodeStatsLogger(log_path=log_file_path)
                checkpoint_callback = CheckpointCallback(
                    save_freq=20000,
                    save_path=model_save_path,
                    name_prefix="checkpoint",
                    save_replay_buffer=True,
                    save_vecnormalize=True,
                )

                # Start Training
                print(f"TensorBoard log path: {tensorboard_log_path}")
                print(f"Model save path:      {model_save_path}")
                model.learn(
                    total_timesteps=TOTAL_TIMESTEPS,
                    progress_bar=True,
                    callback=[custom_callback, checkpoint_callback]
                )

                # Save Final Model and Normalization Stats
                final_model_path = os.path.join(model_save_path, "final_model")
                final_env_stats_path = os.path.join(model_save_path,
                                                    "vec_normalize.pkl")
                model.save(final_model_path)
                env.save(final_env_stats_path)
                print(f"\n--- Final model and stats saved to "
                      f"'{model_save_path}' ---")

            except Exception as e:
                print(f"\n[ERROR] An unexpected error occurred during "
                      f"training for {model_type} with noise {noise}: {e}")
            finally:
                if env:
                    env.close()
