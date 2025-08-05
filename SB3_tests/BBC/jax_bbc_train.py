import os
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from jax_bbc_env import JAXBuckBoostConverterEnv

class EpisodeStatsLogger(CheckpointCallback):
    def __init__(self, log_path, **kwargs):
        super().__init__(**kwargs)
        self.log_path = log_path
        self.log_file = None
        self.rewards = []
        self.lengths = []
        self.current_reward = 0.0
        self.current_length = 0

    def _on_training_start(self) -> None:
        self.log_file = open(self.log_path, "w")
        self.log_file.write("Episode,Total Reward,Length\n")

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        self.current_length += 1
        if self.locals["dones"][0]:
            self.rewards.append(self.current_reward)
            self.lengths.append(self.current_length)
            episode = len(self.rewards)
            self.log_file.write(f"{episode},{self.current_reward:.4f},{self.current_length}\n")
            self.log_file.flush()
            self.current_reward = 0.0
            self.current_length = 0
        return True

    def _on_training_end(self) -> None:
        self.log_file.write("Training completed.\n")
        self.log_file.close()

if __name__ == "__main__":
    MODEL = 'SAC'
    EP_TIME = 0.1
    TIMESTEPS = 100_000
    base = f"./jax_bbc_models/{MODEL}/"
    os.makedirs(base, exist_ok=True)

    log_file = os.path.join(base, f"{MODEL.lower()}_training_log.txt")
    tensorboard_log = os.path.join(base, f"{MODEL.lower()}_tensorboard_log/")

    env = DummyVecEnv([lambda: JAXBuckBoostConverterEnv(max_episode_time=EP_TIME)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        learning_starts=10_000,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
        verbose=1,
        tensorboard_log=tensorboard_log
    )

    logger_cb = EpisodeStatsLogger(log_path=log_file, save_freq=20_000, save_path=base, name_prefix="sac_bbc")
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, callback=logger_cb)

    model.save(os.path.join(base, f"{MODEL.lower()}_bbc_model_final"))
    env.save(os.path.join(base, f"{MODEL.lower()}_vec_normalize_final.pkl"))
    env.close()
