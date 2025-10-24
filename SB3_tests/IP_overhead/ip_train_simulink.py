"""Training script for Simulink-based inverted pendulum environment.

Uses Stable-Baselines3 algorithms (TD3, SAC, DDPG, PPO, A2C, DQN)
to train directly against the Simulink model via MATLAB engine.
Includes TensorBoard logging, checkpointing, and replay buffer saving.
"""

import os
import json
import argparse
import numpy as np
import torch.nn as nn
import time
from tqdm import tqdm


import pandas as pd
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator


from ip_simulink_env import SimulinkEnv, DiscretizedActionWrapper
from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# === Config ===
activation_fn_map = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}

algo_map = {"td3": TD3, "a2c": A2C, "sac": SAC, "ddpg": DDPG, "ppo": PPO, "dqn": DQN}

OFF_POLICY_ALGOS = ["td3", "sac", "ddpg"]


# === Custom Callback ===
class FancyTensorboardCallback(BaseCallback):
    """
    Custom callback for logging and checkpointing during training.

    At specified timesteps, saves model weights and replay buffer (if
    available), logs reward/episode length to TensorBoard, and tracks
    elapsed time between checkpoints.

    Args:
        save_steps (Iterable[int]): Timesteps at which to checkpoint.
        save_path_prefix (str): Path prefix for saved models.
        log_dir (str): TensorBoard logging directory.
        verbose (int): Verbosity level for BaseCallback.

    Attributes:
        saved_steps (set): Timesteps already checkpointed.
        timings (dict): Maps checkpoints to duration (seconds).
        writer (SummaryWriter): TensorBoard summary writer.
    """

    def __init__(self, save_steps, save_path_prefix, log_dir, algo_name=None, verbose=0):
        super().__init__(verbose)
        self.save_steps = sorted(save_steps)
        self.save_path_prefix = save_path_prefix
        self.saved_steps = set()
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
        self.timings = {}
        self.start_time = None
        self.pbar = None
        self.log_dir = log_dir  # keep for export
        self.algo_name = algo_name


    def _on_training_start(self) -> None:
        """Initialize progress bar, timers, and bookkeeping at training start."""
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.total_timesteps = self.model._total_timesteps
        self.pbar = tqdm(
            total=self.total_timesteps, desc="Training Progress", dynamic_ncols=True
        )

    def _on_step(self) -> bool:
        """
        Handle logic at each training step.

        Performs checkpoint saving when a save step is reached,
        logs episode rewards/lengths, and updates timing metrics.

        Returns:
            bool: True to continue training.
        """

        current_time = time.time()
        self.pbar.update(1)

        # Checkpoint saving
        if (
            self.num_timesteps in self.save_steps
            and self.num_timesteps not in self.saved_steps
        ):
            self.model.logger.dump(self.num_timesteps)
            base_dir = os.path.dirname(self.save_path_prefix)
            model_file = os.path.join(base_dir, f"best_model_{self.num_timesteps}.zip")
            buffer_file = os.path.join(
                base_dir, f"replay_buffer_{self.num_timesteps}.pkl"
            )
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
        # Log env timings if the env attached them in info["timings"]
        for info in self.locals.get("infos", []):
            tdict = info.get("timings")
            if not tdict:
                continue
            # Send each scalar timing to TB
            for k, v in tdict.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"env_timings/{k}", v, self.num_timesteps)
            # # Occasionally print a compact line to stdout
            # if self.num_timesteps % 100 == 0:
            #     keys = ("sim_py", "fast_off", "fast_on", "eval_out_angle", "eval_out_vel", "eval_out_time")
            #     msg = ", ".join(f"{k}={tdict[k]:.6f}" for k in keys if k in tdict)
            #     # if msg:
            #         # print(f"[timings @{self.num_timesteps}] {msg}")

        # Log episode reward/length
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.writer.add_scalar(
                    "charts/episode_reward", info["episode"]["r"], self.num_timesteps
                )
                self.writer.add_scalar(
                    "charts/episode_length", info["episode"]["l"], self.num_timesteps
                )

        return True
    
    def _export_tb_scalars_to_csv(self, out_dir: str):
        """Read all TensorBoard scalars from self.log_dir and write one CSV."""
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"tb_scalars_{self.algo_name}.csv")

        ea = event_accumulator.EventAccumulator(self.log_dir)
        ea.Reload()  # parses all event files in the log dir

        rows = []
        for tag in ea.Tags().get("scalars", []):
            for ev in ea.Scalars(tag):
                rows.append({
                    "tag": tag,
                    "step": ev.step,
                    "wall_time": ev.wall_time,
                    "value": ev.value,
                })

        if rows:
            df = pd.DataFrame(rows).sort_values(["tag", "step"])
            df.to_csv(csv_path, index=False)
            print(f"[TB→CSV] Wrote scalar logs to: {csv_path}")
        else:
            print("[TB→CSV] No scalar tags found in TensorBoard logs.")

    def _on_training_end(self):
        """Close progress bar, summarize training times, and flush logs."""
        total_time = time.time() - self.start_time
        self.pbar.close()
        print("\n Training Time Summary:")
        for step in self.save_steps:
            if step in self.timings:
                print(f"    {step} steps: {self.timings[step]:.2f} sec")
        print(f"    Total training time: {total_time:.2f} sec")
        self.writer.flush()
        self.writer.close()
        self._export_tb_scalars_to_csv(out_dir="training_timings")

# === Load Parameters ===
def load_hyperparameters(algo_name):
    """
    Load tuned hyperparameters for given algorithm from JSON.

    Handles PPO special case where batch size may be stored as an index.

    Args:
        algo_name (str): Algorithm key ("ppo", "td3", etc.).

    Returns:
        dict: Dictionary of hyperparameters for SB3 model.
    """

    path = f"jax_hp_results/{algo_name}_best_params.json"
    with open(path, "r") as f:
        params = json.load(f)["best_params"]

    # --- Reconstruct PPO batch-size if it was saved as an index ---
    if algo_name == "ppo" and "batch_size" not in params:
        n_steps = params["n_steps"]
        valid = [i for i in range(32, min(n_steps + 1, 513)) if n_steps % i == 0]
        idx = params.get("batch_size_idx", 0) % len(valid)  # fallback idx = 0
        params["batch_size"] = valid[idx]

    return params


def create_policy_kwargs(params):
    """
    Build SB3 policy_kwargs dictionary from hyperparameters.

    Args:
        params (dict): Hyperparameter dictionary (layer_size, n_layers, etc.).

    Returns:
        dict: policy_kwargs for SB3 algorithms.
    """

    return dict(
        net_arch=[params["layer_size"]] * params["n_layers"],
        activation_fn=activation_fn_map[params["activation_fn"].lower()],
    )


# === Main ===
def main(algo_name="td3", timesteps=100000):
    """
    Train a selected algorithm on the Simulink pendulum environment.

    Creates a fresh Simulink model instance, optionally wraps it for
    discrete actions (DQN), loads hyperparameters, builds model (or
    loads from checkpoint), trains for timesteps, and saves model and
    replay buffer if applicable.

    Args:
        algo_name (str): Algorithm name ("td3", "ppo", "dqn", etc.).
        timesteps (int): Total training timesteps to run.
    """

    # create the raw Simulink env
    base_env = SimulinkEnv(
    model_name="pendulum",
    dt=0.01,
    seed=42,
    # profile=True,                           # turn on instrumentation
    # profile_csv_path="ip_simulink_profile.csv",  # optional, pick your path
    )


    if algo_name == "dqn":
        # define the discrete force‐levels you tuned
        force_values = np.linspace(-2.0, 2.0, 21, dtype=np.float32)
        # wrap it!
        env = DiscretizedActionWrapper(base_env, force_values)
        print("DQN WRAPPED!")
    else:
        env = base_env

    assert algo_name in algo_map, f"Algorithm must be one of: {list(algo_map.keys())}"

    model_base_dir = os.path.join("models", algo_name)
    os.makedirs(model_base_dir, exist_ok=True)
    model_path = os.path.join(model_base_dir, "best_model")
    replay_buffer_path = os.path.join(model_base_dir, "best_model_replay_buffer")
    tensorboard_log_dir = os.path.join("logs", algo_name)

    print(f"Saving models to: {model_base_dir}")
    print(f"TensorBoard logs to: {tensorboard_log_dir}")

    # env = SimulinkEnv(model_name="PendCart", agent_block="PendCart/RL Agent", dt=0.01)
    params = load_hyperparameters(algo_name)
    policy_kwargs = create_policy_kwargs(params)
    Algo = algo_map[algo_name]

    action_noise = None
    if algo_name in ["td3", "ddpg"]:
        action_noise = NormalActionNoise(
            mean=np.zeros(1), sigma=params["action_noise_sigma"] * np.ones(1)
        )

    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}.zip...")
        model = Algo.load(
            model_path,
            env=env,
            action_noise=action_noise if algo_name in ["td3", "ddpg"] else None,
            tensorboard_log=tensorboard_log_dir,
        )
        if algo_name in OFF_POLICY_ALGOS and os.path.exists(
            replay_buffer_path + ".pkl"
        ):
            print("Loading replay buffer...")
            model.load_replay_buffer(replay_buffer_path)
    else:
        print(f"Creating new model for {algo_name.upper()}...")
        common_kwargs = dict(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log_dir,
        )

        if algo_name == "td3":
            model = Algo(
                **common_kwargs,
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                train_freq=(1, "step"),
                # policy_delay=params["policy_delay"],
                action_noise=action_noise,
                # target_policy_noise=params["target_policy_noise"],
                # target_noise_clip=params["target_noise_clip"],
            )
        elif algo_name == "sac":
            model = Algo(
                **common_kwargs,
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                ent_coef=params["ent_coef"],
                train_freq=(1, "step"),
            )
        elif algo_name == "ddpg":
            model = Algo(
                **common_kwargs,
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                action_noise=action_noise,
            )
        elif algo_name == "a2c":
            model = Algo(
                **common_kwargs,
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                rms_prop_eps=params["rms_prop_eps"],
                use_rms_prop=params["use_rms_prop"],
                device="cpu",
            )
        elif algo_name == "ppo":
            model = Algo(
                **common_kwargs,
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                n_epochs=params["n_epochs"],
                clip_range=params["clip_range"],
                ent_coef=params["ent_coef"],
                gae_lambda=params["gae_lambda"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                device="cpu",
            )
        elif algo_name == "dqn":
            # force_values = np.linspace(-10.0, 10.0, 11, dtype=np.float32)
            # env = DiscretizedActionWrapper(env, force_values)
            # Off-policy DQN with replay & epsilon-greedy schedule
            model = Algo(
                **common_kwargs,
                buffer_size=params["buffer_size"],
                batch_size=params["batch_size"],
                tau=params["tau"],
                train_freq=(params["train_freq"], "step"),
                target_update_interval=params["target_update_interval"],
                exploration_fraction=params["exploration_fraction"],
                exploration_final_eps=params["exploration_final_eps"],
                learning_starts=5000,  # added because not saved in hyperparams
            )

    # Setup callback
    checkpoint_steps = {10_000, 25_000, 50_000, 75_000, 100_000}
    callback = FancyTensorboardCallback(
        save_steps=checkpoint_steps,
        save_path_prefix=model_path,
        log_dir=tensorboard_log_dir,
        algo_name=algo_name,   # <-- add this line
    )

    print(f"Training {algo_name.upper()} for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=False,
        callback=callback,
        tb_log_name="run",
    )

    model.save(model_path)
    print(f"Final model saved to {model_path}.zip")

    if algo_name in OFF_POLICY_ALGOS:
        model.save_replay_buffer(replay_buffer_path)
        print(f"Final replay buffer saved to {replay_buffer_path}.pkl")

    env.close()
    print("Training complete. Environment closed.")


# === CLI Entry ===
if __name__ == "__main__":
    """
    CLI entry point.

    Parses command-line arguments for algorithm and timesteps, then
    launches training using `main()`.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", choices=["td3", "a2c", "sac", "ddpg", "ppo", "dqn"], default="a2c"
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    args = parser.parse_args()
    main(algo_name=args.algo, timesteps=args.timesteps)
