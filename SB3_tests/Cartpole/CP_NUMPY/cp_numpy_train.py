import os
import json
import argparse
import numpy as np
import torch.nn as nn
import torch
import random
import time
from tqdm import tqdm

from cp_numpy_wrapper import CartPoleGymWrapper, DiscretizedActionWrapper
from stable_baselines3 import TD3, A2C, SAC, DDPG, PPO, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter

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
        """ Initialize checkpointing, timing, and TensorBoard logging.

            Parameters
            ----------
            save_steps : Iterable[int]
                Timesteps at which to save model (and replay buffer if supported).
            save_path_prefix : str
                Prefix path for checkpoint files (e.g., ".../best_model").
            log_dir : str
                Directory for TensorBoard logs.
            verbose : int, optional
                Verbosity level passed to `BaseCallback`.
        """

        super().__init__(verbose)
        self.save_steps = sorted(save_steps)
        self.save_path_prefix = save_path_prefix
        self.saved_steps = set()
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=60)
        self.timings = {}
        self.start_time = None
        self.pbar = None

    def _on_training_start(self) -> None:
        """ Prepare timers and a progress bar at the start of training.

            Initializes timing references, reads the total timesteps from the model,
            and creates a tqdm progress bar for live feedback.
        """

        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.total_timesteps = self.model._total_timesteps
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", dynamic_ncols=True)

    def _on_step(self) -> bool:
        """ Run after each environment step: log, checkpoint, and advance pbar.

            - Updates the progress bar by one step.
            - At configured `save_steps`, saves the model and replay buffer (if
            available) and records elapsed time since the previous checkpoint.
            - Logs per-episode reward and length to TensorBoard when present.

            Returns
            -------
            bool
                Always True to continue training.
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

            print(f"\nCheckpoint at {self.num_timesteps} steps:")
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
        """ Print a timing summary and close logging resources at the end.

            Closes the progress bar, prints per-checkpoint durations and total
            training time, then flushes and closes the TensorBoard writer.
            """

        total_time = time.time() - self.start_time
        self.pbar.close()
        print("\nTraining Time Summary:")
        for step in self.save_steps:
            if step in self.timings:
                print(f"    {step} steps: {self.timings[step]:.2f} sec")
        print(f"    Total training time: {total_time:.2f} sec")
        self.writer.flush()
        self.writer.close()


def load_hyperparameters(algo_name):
    """ Load tuned hyperparameters for an algorithm from JSON.

        Reads `numpy_hp_results/{algo_name}_best_params.json` and returns its
        `"best_params"`. For PPO, if `batch_size` was saved indirectly as an
        index, reconstruct a valid batch size that evenly divides `n_steps`.

        Parameters
        ----------
        algo_name : str
            One of {"td3","a2c","sac","ddpg","ppo","dqn"}.

        Returns
        -------
        dict
            Hyperparameters suitable for initializing the SB3 model.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        json.JSONDecodeError
            If the JSON cannot be parsed.
        KeyError
            If expected keys are missing.
    """

    path = f"numpy_hp_results/{algo_name}_best_params.json"
    with open(path, "r") as f:
        params = json.load(f)["best_params"]

    if algo_name == "ppo" and "batch_size" not in params:
        n_steps = params["n_steps"]
        valid = [i for i in range(32, min(n_steps + 1, 513)) if n_steps % i == 0]
        idx = params.get("batch_size_idx", 0) % len(valid)
        params["batch_size"] = valid[idx]

    return params


def create_policy_kwargs(params):
    """Construct SB3 `policy_kwargs` (MLP depth/width/activation).

        Parameters
        ----------
        params : dict
            Expected keys: {"layer_size", "n_layers", "activation_fn"}.

        Returns
        -------
        dict
            Keyword arguments for SB3 policy creation.
    """

    return dict(
        net_arch=[params["layer_size"]] * params["n_layers"],
        activation_fn=activation_fn_map[params["activation_fn"].lower()]
    )


def main(algo_name="ppo", timesteps=100_000, noise=False, noise_level=0.01):
    """ Train an SB3 algorithm on the NumPy CartPole and save artifacts.

        Sets seeds for reproducibility, builds the Gym wrapper (optional
        Gaussian observation noise), wraps with `DiscretizedActionWrapper` for
        DQN, loads tuned hyperparameters, constructs the selected SB3 algorithm
        (with action noise for TD3/DDPG), trains with a TensorBoard/Checkpoint
        callback, saves the final model (and replay buffer for off-policy algos),
        then closes the environment.

        Parameters
        ----------
        algo_name : str, optional
            One of {"td3","a2c","sac","ddpg","ppo","dqn"}. Default "ppo".
        timesteps : int, optional
            Total timesteps to train for. Default 100_000.
        noise : bool, optional
            If True, add Gaussian noise to observations. Default False.
        noise_level : float, optional
            Standard deviation for observation noise. Default 0.01.

        Returns
        -------
        None
    """

    # Set all seeds for reproducibility
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    # Instantiate environment
    env = CartPoleGymWrapper(seed=SEED, noise=noise, noise_std=noise_level)
    if algo_name == "dqn":
        force_values = np.linspace(-10.0, 10.0, 5)
        env = DiscretizedActionWrapper(env, force_values=force_values)

    assert algo_name in algo_map, f"Algorithm must be one of: {list(algo_map.keys())}"

    # Directories
    model_base_dir = os.path.join("numpy", algo_name)
    os.makedirs(model_base_dir, exist_ok=True)
    model_path = os.path.join(model_base_dir, "best_model")
    replay_buffer_path = os.path.join(model_base_dir, "best_model_replay_buffer")
    tensorboard_log_dir = os.path.join("numpy_logs", algo_name)

    print(f"📁 Saving models to: {model_base_dir}")
    print(f"📊 TensorBoard logs to: {tensorboard_log_dir}")

    # Load hyperparameters
    params = load_hyperparameters(algo_name)
    policy_kwargs = create_policy_kwargs(params)
    Algo = algo_map[algo_name]

    action_noise = None
    if algo_name in ["td3", "ddpg"]:
        action_noise = NormalActionNoise(
            mean=np.zeros(1),
            sigma=params["action_noise_sigma"] * np.ones(1)
        )

    # Load or create model
    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}.zip...")
        model = Algo.load(
            model_path,
            env=env,
            action_noise=action_noise if algo_name in OFF_POLICY_ALGOS else None,
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

    # Train
    checkpoint_steps = {10_000, 25_000, 50_000, 75_000, timesteps}
    callback = FancyTensorboardCallback(save_steps=checkpoint_steps,
                                       save_path_prefix=model_path,
                                       log_dir=tensorboard_log_dir)

    print(f"Training {algo_name.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False,
                callback=callback, tb_log_name="run")

    # Save final model and buffer
    model.save(model_path)
    print(f"Final model saved to {model_path}.zip")
    if algo_name in OFF_POLICY_ALGOS:
        model.save_replay_buffer(replay_buffer_path)
        print(f"Final replay buffer saved to {replay_buffer_path}.pkl")

    env.close()
    print("Training complete. Environment closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=list(algo_map.keys()) + ["all"],
        default=["ppo"],
        help="Which algorithm(s) to train; use 'all' to run every algo in sequence."
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--noise", action="store_true", help="Add Gaussian noise to observations")
    parser.add_argument("--noise-level", type=float, default=0.01, help="Standard deviation for observation noise")
    args = parser.parse_args()

    # Expand "all" into the full list
    if "all" in args.algos:
        algos_to_run = list(algo_map.keys())
    else:
        algos_to_run = args.algos

    for algo in algos_to_run:
        print(f"\n Starting training for {algo.upper()} …")
        main(
            algo_name=algo,
            timesteps=args.timesteps,
            noise=args.noise,
            noise_level=args.noise_level,
        )
