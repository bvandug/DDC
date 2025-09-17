"""Training script for NumPy-based inverted pendulum environment.

Supports multiple SB3 algorithms (TD3, SAC, DDPG, PPO, A2C, DQN),
TensorBoard logging, replay buffer saving/loading, checkpointing,
and deterministic seeding for reproducibility.
"""
import os
import json
import argparse
import numpy as np
import torch.nn as nn
import time
from tqdm import tqdm
import random
import torch

from ip_numpy_wrapper import InvertedPendulumGymWrapper
from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.utils import set_random_seed

activation_fn_map = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU
}

algo_map = {
    "td3": TD3,
    "a2c": A2C,
    "sac": SAC,
    "ddpg": DDPG,
    "ppo": PPO,
    "dqn": DQN
}

OFF_POLICY_ALGOS = ["td3", "sac", "ddpg"]

class FancyTensorboardCallback(BaseCallback):
    """Custom callback for periodic checkpointing and TensorBoard logging.

    At specified timesteps, saves model and replay buffer (if available),
    logs rewards and episode lengths to TensorBoard, and tracks elapsed
    time between checkpoints.

    Args:
        save_steps (Iterable[int]): Timesteps at which to checkpoint.
        save_path_prefix (str): Base path for model and buffer files.
        log_dir (str): Directory for TensorBoard logs.
        verbose (int): Verbosity level.

    Attributes:
        saved_steps (set): Set of steps already checkpointed.
        timings (dict): Maps checkpoint steps to durations (seconds).
        writer (SummaryWriter): TensorBoard log writer.
    """
    def __init__(self, save_steps, save_path_prefix, log_dir, verbose=0):
        super().__init__(verbose)
        self.save_steps = sorted(save_steps)
        self.save_path_prefix = save_path_prefix
        self.saved_steps = set()
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
        self.timings = {}
        self.start_time = None
        self.pbar = None

    def _on_training_start(self) -> None:
        """Initialise progress bar, timer, and internal counters at start."""
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.total_timesteps = self.model._total_timesteps
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", dynamic_ncols=True)

    def _on_step(self) -> bool:
        """
        Execute callback at each training step.

        Checks if current timestep matches a checkpoint, saves model/buffer,
        logs episode metrics to TensorBoard, and updates timing stats.

        Returns:
            bool: True to continue training.
        """
        current_time = time.time()
        self.pbar.update(1)

        if self.num_timesteps in self.save_steps and self.num_timesteps not in self.saved_steps:
            self.model.logger.dump(self.num_timesteps)
            base_dir = os.path.dirname(self.save_path_prefix)
            model_file = os.path.join(base_dir, f"best_model_{self.num_timesteps}.zip")
            buffer_file = os.path.join(base_dir, f"replay_buffer_{self.num_timesteps}.pkl")
            self.writer.flush()

            self.model.save(model_file)
            if hasattr(self.model, "save_replay_buffer"):
                self.model.save_replay_buffer(buffer_file)

            duration = current_time - self.last_check_time
            self.timings[self.num_timesteps] = duration
            self.last_check_time = current_time

            print(f"\n Checkpoint at {self.num_timesteps} steps:")
            print(f"    - Model: {model_file}")
            print(f"    - Replay buffer: {buffer_file}")
            print(f"    - Elapsed: {duration:.2f} sec")

            self.saved_steps.add(self.num_timesteps)

        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.writer.add_scalar("charts/episode_reward", info["episode"]["r"], self.num_timesteps)
                self.writer.add_scalar("charts/episode_length", info["episode"]["l"], self.num_timesteps)

        return True

    def _on_training_end(self):
        """Close progress bar, print timing summary, and flush/close writer."""
        total_time = time.time() - self.start_time
        self.pbar.close()
        print("\n Training Time Summary:")
        for step in self.save_steps:
            if step in self.timings:
                print(f"    {step} steps: {self.timings[step]:.2f} sec")
        print(f"    Total training time: {total_time:.2f} sec")
        self.writer.flush()
        self.writer.close()

def load_hyperparameters(algo_name):
    """Load best hyperparameters for a given algorithm from JSON.

    Handles PPO special case where batch size may be stored as an index.

    Args:
        algo_name (str): Algorithm key (e.g., "ppo", "td3").

    Returns:
        dict: Dictionary of hyperparameters for SB3 model.
    """
    path = f"jax_hp_results/{algo_name}_best_params.json"
    with open(path, "r") as f:
        params = json.load(f)["best_params"]

    if algo_name == "ppo" and "batch_size" not in params:
        n_steps = params["n_steps"]
        valid = [i for i in range(32, min(n_steps + 1, 513)) if n_steps % i == 0]
        idx = params.get("batch_size_idx", 0) % len(valid)
        params["batch_size"] = valid[idx]

    return params

def create_policy_kwargs(params):
    """
    Build SB3 policy_kwargs dictionary from hyperparameter set.

    Args:
        params (dict): Hyperparameters including architecture size and activation.

    Returns:
        dict: policy_kwargs for SB3 algorithms.
    """
    return dict(
        net_arch=[params["layer_size"]] * params["n_layers"],
        activation_fn=activation_fn_map[params["activation_fn"].lower()]
    )

def main(algo_name="ppo", timesteps=100_000):
    """
    Train an RL algorithm on the NumPy-based pendulum environment.

    Creates environment, loads hyperparameters, builds model (or loads
    existing checkpoint), sets up logging and checkpointing, trains
    for specified timesteps, and saves final model and replay buffer.

    Args:
        algo_name (str): Algorithm name ("ppo", "td3", etc.).
        timesteps (int): Number of timesteps to train.
    """

    SEED = 42

    # Set seeds globally
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    env = InvertedPendulumGymWrapper(seed=SEED)

    assert algo_name in algo_map, f"Algorithm must be one of: {list(algo_map.keys())}"

    model_base_dir = os.path.join("ip_numpy_models", algo_name)
    os.makedirs(model_base_dir, exist_ok=True)
    model_path = os.path.join(model_base_dir, "best_model")
    replay_buffer_path = os.path.join(model_base_dir, "best_model_replay_buffer")
    tensorboard_log_dir = os.path.join("ip_numpy_train_logs", algo_name)

    print(f"Saving models to: {model_base_dir}")
    print(f"TensorBoard logs to: {tensorboard_log_dir}")

    params = load_hyperparameters(algo_name)
    policy_kwargs = create_policy_kwargs(params)
    Algo = algo_map[algo_name]

    action_noise = None
    if algo_name in ["td3", "ddpg"]:
        action_noise = NormalActionNoise(
            mean=np.zeros(1),
            sigma=params["action_noise_sigma"] * np.ones(1)
        )

    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}.zip...")
        model = Algo.load(
            model_path,
            env=env,
            action_noise=action_noise if algo_name in ["td3", "ddpg"] else None,
            tensorboard_log=tensorboard_log_dir
        )
        if algo_name in OFF_POLICY_ALGOS and os.path.exists(replay_buffer_path + ".pkl"):
            print("Loading replay buffer...")
            model.load_replay_buffer(replay_buffer_path)
    else:
        print(f"Creating new model for {algo_name.upper()}...")
        common_kwargs = dict(
            policy="MlpPolicy",
            env=env,
            seed=SEED,
            verbose=1,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log_dir
        )

        if algo_name == "td3":
            model = Algo(**common_kwargs,
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                train_freq=(1, "step"),
                #policy_delay=params["policy_delay"],
                action_noise=action_noise
                #,
                #target_policy_noise=params["target_policy_noise"],
                #target_noise_clip=params["target_noise_clip"]
                )
        elif algo_name == "sac":
            model = Algo(**common_kwargs,
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                ent_coef=params["ent_coef"],
                train_freq=(1, "step"))
        elif algo_name == "ddpg":
            model = Algo(**common_kwargs,
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                action_noise=action_noise)
        elif algo_name == "a2c":
            model = Algo(**common_kwargs,
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                rms_prop_eps=params["rms_prop_eps"],
                use_rms_prop=params["use_rms_prop"],
                device="cpu")
        elif algo_name == "ppo":
            model = Algo(**common_kwargs,
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                n_epochs=params["n_epochs"],
                clip_range=params["clip_range"],
                ent_coef=params["ent_coef"],
                gae_lambda=params["gae_lambda"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                device="cpu")
        elif algo_name == "dqn":
            model = Algo(**common_kwargs,
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                train_freq=(params["train_freq"], "step"),
                target_update_interval=params["target_update_interval"],
                exploration_fraction=params["exploration_fraction"],
                exploration_final_eps=params["exploration_final_eps"],
                learning_starts=5000)

    checkpoint_steps = {10_000, 25_000, 50_000, 75_000, timesteps}
    callback = FancyTensorboardCallback(
        save_steps=checkpoint_steps,
        save_path_prefix=model_path,
        log_dir=tensorboard_log_dir
    )

    print(f"Training {algo_name.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False,
                callback=callback, tb_log_name="run")

    model.save(model_path)
    print(f"Final model saved to {model_path}.zip")

    if algo_name in OFF_POLICY_ALGOS:
        model.save_replay_buffer(replay_buffer_path)
        print(f"Final replay buffer saved to {replay_buffer_path}.pkl")

    env.close()
    print("Training complete. Environment closed.")

if __name__ == "__main__":
    """CLI entry point.

    Iterates through a predefined list of algorithms and trains each
    for 100k timesteps using the same training pipeline.
    """
    for algo in ["td3","ddpg","sac","ppo","a2c"]:
        main(algo_name=algo, timesteps=100_000)
