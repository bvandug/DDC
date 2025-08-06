import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from np_bbc_env import JAXBuckBoostConverterEnv

from stable_baselines3.common.callbacks import BaseCallback

class EpisodeStatsLogger(BaseCallback):
    def __init__(self, log_path, **kwargs):
        super().__init__(**kwargs)
        self.log_path = log_path
        self.log_file = None
        self.rewards = []
        self.lengths = []
        self.current_reward = 0.0
        self.current_length = 0
        self.pbar = None

    def _on_training_start(self) -> None:
        self.log_file = open(self.log_path, "w")
        self.log_file.write("Episode,Total Reward,Length\n")
        # if you still want tqdm, leave this; otherwise you can drop it
        # from tqdm import tqdm
        # total = self.model._total_timesteps
        # self.pbar = tqdm(total=total, desc="Training Steps")

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]
        self.current_reward += reward
        self.current_length += 1

        # advance tqdm if you’re using it
        if self.pbar:
            self.pbar.update(1)

        if done:
            ep = len(self.rewards) + 1
            self.rewards.append(self.current_reward)
            self.lengths.append(self.current_length)

            # build the line once
            line = f"{ep},{self.current_reward:.4f},{self.current_length}"
            # write to file
            self.log_file.write(line + "\n")
            self.log_file.flush()
            # **also print to console**
            print(line)

            # reset counters
            self.current_reward = 0.0
            self.current_length = 0

        return True

    def _on_training_end(self) -> None:
        self.log_file.write("Training completed.\n")
        self.log_file.close()
        if self.pbar:
            self.pbar.close()


if __name__ == "__main__":
    MODEL = 'A2C'
    TIMESTEPS = 300_000
    base = f"./numpy_bbc_models/{MODEL}/"
    os.makedirs(base, exist_ok=True)

    log_file = os.path.join(base, f"{MODEL.lower()}_training_log.txt")
    tensorboard_log = os.path.join(base, f"{MODEL.lower()}_tensorboard_log/")

    env = DummyVecEnv([lambda: JAXBuckBoostConverterEnv(
        max_episode_steps=2000,
        frame_skip=10,
        grace_period_steps=50,
        dt=5e-6,
        target_voltage=-30.0
    )])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = A2C(
    "MlpPolicy", env,
    learning_rate=1e-4,        # lower LR for more stable convergence
    n_steps=100,                 # smaller rollout for faster feedback
    gamma=0.995,               # slightly higher discount for long‐term view
    gae_lambda=0.95,           # GAE smoothing
    ent_coef=0.02,             # stronger entropy bonus to keep exploring
    vf_coef=0.5,               # balance value loss
    max_grad_norm=0.5,         # gradient clipping
    # rms_prop_eps = 2.2032810428869103e-05,
    # use_rms_prop = True,
    policy_kwargs=dict(net_arch=[128, 128, 128]),  # a bit smaller network
    verbose=0,                 # enable SB3 console logs
    tensorboard_log=tensorboard_log,
)

    cb = EpisodeStatsLogger(log_path=log_file)
    model.learn(
    total_timesteps=TIMESTEPS,
    callback=cb,
    progress_bar=False,
    log_interval=10,
)

    model.save(os.path.join(base, f"{MODEL.lower()}_bbc_model_final"))
    env.save(os.path.join(base, f"{MODEL.lower()}_vec_normalize_final.pkl"))
    env.close()
