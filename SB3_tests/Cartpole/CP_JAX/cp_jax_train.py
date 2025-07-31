# ip_jax_train.py (seeded version)
import os
import json
import argparse
import numpy as np
import torch.nn as nn
import time
from tqdm import tqdm
import random

from cp_jax_wrapper import CartPoleGymWrapper, DiscretizedActionWrapper
from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

import torch
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
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.total_timesteps = self.model._total_timesteps
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", dynamic_ncols=True)

    def _on_step(self) -> bool:
        current_time = time.time()
        self.pbar.update(1)

        if self.num_timesteps in self.save_steps and self.num_timesteps not in self.saved_steps:
            self.model.logger.dump(self.num_timesteps)
            base_dir = os.path.dirname(self.save_path_prefix)
            model_file = os.path.join(base_dir, f"{self.save_path_prefix.split('/')[-1]}_{self.num_timesteps}.zip")
            buffer_file = os.path.join(base_dir, f"{self.save_path_prefix.split('/')[-1]}_replay_buffer_{self.num_timesteps}.pkl")
            self.writer.flush()

            self.model.save(model_file)
            if hasattr(self.model, "save_replay_buffer"):
                self.model.save_replay_buffer(buffer_file)

            duration = current_time - self.last_check_time
            self.timings[self.num_timesteps] = duration
            self.last_check_time = current_time

            print(f"\n📌 Checkpoint at {self.num_timesteps} steps:")
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
        total_time = time.time() - self.start_time
        self.pbar.close()
        print("\n🕒 Training Time Summary:")
        for step in self.save_steps:
            if step in self.timings:
                print(f"    {step} steps: {self.timings[step]:.2f} sec")
        print(f"    Total training time: {total_time:.2f} sec")
        self.writer.flush()
        self.writer.close()


def load_hyperparameters(algo_name):
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
    return dict(
        net_arch=[params["layer_size"]] * params["n_layers"],
        activation_fn=activation_fn_map[params["activation_fn"].lower()]
    )


def main(algo_name="ppo",
         timesteps=100_000,
         noise: bool = False,
         noise_level: float = 0.01):
    SEED = 42

    # Set all seeds
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    # Determine prefix
    prefix = f"noise_{noise_level:.3f}_" if noise else ""

    # Create and seed the environment
    env = CartPoleGymWrapper(
        seed=SEED,
        noise=noise,
        noise_std=noise_level
    )

    # If using DQN (or similar), wrap with DiscretizedActionWrapper
    if algo_name == "dqn":
        force_values = np.linspace(-10.0, 10.0, 5)
        env = DiscretizedActionWrapper(env, force_values=force_values)

    # Directories and file names
    model_base_dir     = os.path.join("jax", prefix + algo_name)
    os.makedirs(model_base_dir, exist_ok=True)
    model_path         = os.path.join(model_base_dir, prefix + "best_model")
    replay_buffer_path = os.path.join(model_base_dir, prefix + "best_model_replay_buffer")
    tensorboard_log_dir = os.path.join("jax_logs", prefix + algo_name)

    print(f"📁 Saving models to: {model_base_dir}")
    print(f"📊 TensorBoard logs to: {tensorboard_log_dir}")

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
                         action_noise=action_noise
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

    print(f"🚀 Training {algo_name.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False,
                callback=callback, tb_log_name="run")

    model.save(model_path)
    print(f"✅ Final model saved to {model_path}.zip")

    if algo_name in OFF_POLICY_ALGOS:
        model.save_replay_buffer(replay_buffer_path)
        print(f"✅ Final replay buffer saved to {replay_buffer_path}.pkl")

    env.close()
    print("🏁 Training complete. Environment closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=list(algo_map.keys()) + ["all"],
        default=["ppo"],
        help="Which algorithm(s) to train; use 'all' to run every algo in sequence.",
    )
    parser.add_argument("--timesteps",    type=int,   default=100_000)
    parser.add_argument("--noise",        action="store_true", help="…")
    parser.add_argument("--noise-level",  type=float, default=0.01,    help="…")
    args = parser.parse_args()

    # Expand "all" into the full list
    if "all" in args.algos:
        algos_to_run = list(algo_map.keys())
    else:
        algos_to_run = args.algos

    for algo in algos_to_run:
        print(f"\n🔄 Starting training for {algo.upper()} …")
        main(
            algo_name=algo,
            timesteps=args.timesteps,
            noise=args.noise,
            noise_level=args.noise_level,
        )

