# jax_bbc_train.py — Buck-Boost training (A2C, SAC, or TD3) with vectorized envs

import os
import argparse
import numpy as np
import torch.nn as nn
import json
import shutil
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import A2C, SAC, TD3, DQN, PPO, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from np_bbc_env import JAXBuckBoostConverterEnv
import numpy as np
activation_fn_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}

# === IP-style HP loading (simple helpers, JSON is source of truth) ===
def load_hyperparameters(algo_name: str):
    path = os.path.join("bbc_17_hp_results", f"{algo_name.lower()}_best_params.json")
    with open(path, "r") as f:
        blob = json.load(f)
    # tuner saves {"best_params": {...}}; support flat dict too
    return blob.get("best_params", blob)

def create_policy_kwargs(params: dict, algo: str | None = None):
    net = [params["layer_size"]] * int(params["n_layers"])
    act = activation_fn_map[params["activation_fn"].lower()]
    if (algo or "").lower() == "a2c":
        # A2C in this script uses separate pi/vf nets; others use shared list
        return dict(net_arch=dict(pi=net, vf=net), activation_fn=act)
    return dict(net_arch=net, activation_fn=act)



# ===== Discretize a 1-D continuous duty command into N fixed levels (for DQN) =====
class DiscretizedDutyWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, duty_levels: np.ndarray):
        super().__init__(env)
        duty_levels = np.asarray(duty_levels, dtype=np.float32)
        assert duty_levels.ndim == 1 and duty_levels.size >= 2
        self.duty_levels = duty_levels
        self.action_space = spaces.Discrete(duty_levels.size)

    def action(self, act: int):
        d = float(self.duty_levels[int(act)])
        return np.array([d], dtype=np.float32)

class EpisodeStatsLogger(BaseCallback):
    """
    Logs raw per-episode stats emitted by Monitor:
      - TotalReward (unsmoothed)
      - Length (steps)

    Also writes the same raw values to TensorBoard under 'custom/*'.
    """
    def __init__(self, log_path: str, log_tensorboard: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.log_path = log_path
        self.log_tensorboard = log_tensorboard
        self.log_file = None
        self.ep_idx = 0

    def _on_training_start(self) -> None:
        # line-buffered so rows appear immediately
        self.log_file = open(self.log_path, "w", buffering=1)
        self.log_file.write("Episode,TotalReward,Length,AvgDuty,AvgVoltage\n")


    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue

            self.ep_idx += 1
            total_r = float(ep.get("r", np.nan))
            length  = int(ep.get("l", 0))
            avg_duty = float(ep.get("avg_duty", np.nan))
            avg_voltage = float(ep.get("avg_voltage", np.nan))
            # CSV row
            self.log_file.write(f"{self.ep_idx},{total_r:.6f},{length},{avg_duty:.6f},{avg_voltage:.6f}\n")

            # Optional: raw values in TensorBoard
            if self.log_tensorboard:
                self.logger.record("custom/ep_reward_raw", total_r)
                self.logger.record("custom/ep_len_raw", length)
                self.logger.record("custom/ep_avg_duty_raw", avg_duty)
                self.logger.record("custom/ep_avg_voltage_raw", avg_voltage)
        return True

    def _on_training_end(self) -> None:
        if self.log_file:
            self.log_file.write("Training completed.\n")
            self.log_file.flush()
            self.log_file.close()


# ===== Macro-step wrapper: repeat the same action for k PWM periods =====
class MultiPeriodStep(gym.Wrapper):
    def __init__(self, env, k: int = 1):
        super().__init__(env)
        assert k >= 1
        self.k = k

    def step(self, action):
        total_r = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self.k):
            obs, r, t, tr, info = self.env.step(action)
            total_r += float(r)
            terminated |= bool(t)
            truncated  |= bool(tr)
            if terminated or truncated:
                break
        return obs, total_r, terminated, truncated, info

def duty_bins_uniform(dmin=0.10, dmax=0.90, n=17):
    return np.linspace(dmin, dmax, n, dtype=np.float32)


def make_env(seed: int, rank: int = 0, k_macro: int = 1,
             algo_name: str = "td3",
             dqn_bins: int = 17,
             duty_min: float = 0.10,
             duty_max: float = 0.90,
             voltage_noise_std: float = 0.0):
    def _thunk():
        e = JAXBuckBoostConverterEnv(
            dt=5e-6,
            frame_skip=26,
            max_episode_steps=4000,
            grace_period_steps=200,
            target_voltage=-80.0,
            voltage_noise_std=voltage_noise_std,
        )
        if hasattr(e, "_clip_low"):  e._clip_low  = -10.0
        if hasattr(e, "_clip_high"): e._clip_high =  10.0

        if k_macro > 1:
            e = MultiPeriodStep(e, k=k_macro)

        # --- Discretize duty for DQN ---
        if algo_name.lower() == "dqn":
            # Uniform bins over [duty_min, duty_max]; swap to equal_output_ratio if you prefer
            levels = duty_bins_uniform(duty_min, duty_max, dqn_bins)
            e = DiscretizedDutyWrapper(e, duty_levels=levels)

        e = Monitor(e, info_keywords=("iL", "vC", "mag_vC", "e_norm", "dduty", "in_band",
                                      "avg_duty", "avg_voltage"))
        e.reset(seed=seed + rank)
        return e
    return _thunk



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["a2c", "sac", "td3", "dqn", "ppo", "ddpg"], default="a2c")
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel envs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--macro-k", type=int, default=1, help="Repeat each action for k PWM periods")
    # Arguments for randomizing input voltage
    parser.add_argument("--randomize-vin", action="store_true", help="Randomize input voltage each episode")
    parser.add_argument("--vin-min", type=float, default=46.5, help="Min input voltage for randomization")
    parser.add_argument("--vin-max", type=float, default=49.5, help="Max input voltage for randomization")

    parser.add_argument("--dqn-bins", type=int, default=17, help="Number of duty bins for DQN")
    parser.add_argument("--duty-min", type=float, default=0.10, help="Min duty for bins")
    parser.add_argument("--duty-max", type=float, default=0.90, help="Max duty for bins")
    parser.add_argument("--voltage-noise-std", type=float, default=0.0, help="Std dev of Gaussian noise to add to voltage observation")
    parser.add_argument(
        "--noise",
        type=str,
        default="single",
        choices=["single", "all"],
        help="Noise mode: 'single' (use --voltage-noise-std) or 'all' (train across [0, 0.1, 0.01, 0.001])"
    )

    args = parser.parse_args()

    # Load best params (single source of truth, like IP)
    params = load_hyperparameters(args.algo)

    # === BEGIN minimal addition for sequential `--noise all` ===
    def _train_one_noise(nl: float):
        suffix = f"noise_{nl:.3f}"
        model_base_dir = os.path.join("jax_models_80", f"{args.algo}_{suffix}")
        os.makedirs(model_base_dir, exist_ok=True)
        model_path = os.path.join(model_base_dir, "best_model")
        replay_buffer_path = os.path.join(model_base_dir, "best_model_replay_buffer")
        tensorboard_log = os.path.join("jax_train_logs_110", f"{args.algo}_{suffix}")
        os.makedirs(tensorboard_log, exist_ok=True)
        log_file = os.path.join(model_base_dir, f"{args.algo}_training_log.txt")

        print(f"\n=== Training for noise std = {nl:.3f} ===")
        print(f"📁 Model directory: {model_base_dir}")
        print(f"🧠 Model file:      {model_path}.zip")
        print(f"📊 TensorBoard dir: {tensorboard_log}")
        print(f"📝 Episode CSV:     {log_file}")

        # Single-noise vec env
        n_envs = max(1, args.n_envs)
        env_fns = [make_env(args.seed, i, args.macro_k,
                            algo_name=args.algo,
                            dqn_bins=args.dqn_bins,
                            duty_min=args.duty_min,
                            duty_max=args.duty_max,
                            voltage_noise_std=nl) for i in range(n_envs)]
        vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
        env = vec_cls(env_fns)
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

        # Build/resume model using your existing per-algo params
        policy_kwargs = create_policy_kwargs(params, args.algo)
        AlgoMap = {"a2c": A2C, "sac": SAC, "td3": TD3, "dqn": DQN, "ppo": PPO, "ddpg": DDPG}
        model = None

        if os.path.exists(model_path + ".zip"):
            print(f"🔁 Loading existing model from {model_path}.zip")
            model = AlgoMap[args.algo].load(
                model_path, env=env, tensorboard_log=tensorboard_log,
                device=args.device, seed=args.seed
            )
            # Resume replay buffer for off-policy algos
            if args.algo in ["td3","sac","ddpg"] and os.path.exists(replay_buffer_path + ".pkl"):
                try:
                    model.load_replay_buffer(replay_buffer_path)
                except Exception as e:
                    print(f"⚠️ Could not load replay buffer: {e}")
        else:
            if args.algo == "a2c":
                model = A2C(
                "MlpPolicy", env,
                learning_rate=float(params["learning_rate"]),
                n_steps=int(params["n_steps"]),
                gamma=float(params["gamma"]),
                gae_lambda=float(params["gae_lambda"]),
                ent_coef=float(params["ent_coef"]),
                vf_coef=float(params["vf_coef"]),
                max_grad_norm=float(params["max_grad_norm"]),
                normalize_advantage=bool(params["normalize_advantage"]),  # ← add this
                policy_kwargs=policy_kwargs,
                use_rms_prop=bool(params["use_rms_prop"]),
                rms_prop_eps=float(params["rms_prop_eps"]),
                verbose=1, tensorboard_log=tensorboard_log,
                device=args.device, seed=args.seed,
            )

            elif args.algo == "sac":
                model = SAC(
                    "MlpPolicy", env,
                    learning_rate=float(params["learning_rate"]),
                    buffer_size=int(params["buffer_size"]),
                    batch_size=int(params["batch_size"]),
                    tau=float(params["tau"]),
                    gamma=float(params["gamma"]),
                    train_freq=int(params["train_freq"]),
                    gradient_steps=int(params["gradient_steps"]),
                    learning_starts=int(params["learning_starts"]),
                    policy_kwargs=policy_kwargs,
                    verbose=1, tensorboard_log=tensorboard_log,
                    device=args.device, seed=args.seed,
                )
            if args.algo == "td3":
                train_freq = (int(params["train_freq_k"]), "step")
                learning_starts = int(params.get("learning_starts", max(5000, float(params.get("learning_starts_frac", 0)) * int(params["buffer_size"]))))
                act_dim = env.action_space.shape[0]
                sigma = float(params.get("action_noise_sigma", 0.0))
                action_noise = NormalActionNoise(mean=np.zeros(act_dim), sigma=sigma * np.ones(act_dim)) if sigma > 0 else None
                
                # Calculate target_noise_clip from multiplier in JSON
                target_policy_noise = float(params["target_policy_noise"])
                target_noise_clip_mult = float(params["target_noise_clip_mult"])
                target_noise_clip = target_policy_noise * target_noise_clip_mult

                model = TD3("MlpPolicy", env,
                            learning_rate=float(params["learning_rate"]),
                            buffer_size=int(params["buffer_size"]),
                            batch_size=int(params["batch_size"]),
                            tau=float(params["tau"]),
                            gamma=float(params["gamma"]),
                            train_freq=train_freq,
                            gradient_steps=int(params["gradient_steps"]),
                            learning_starts=learning_starts,
                            policy_delay=int(params["policy_delay"]),
                            target_policy_noise=target_policy_noise,
                            target_noise_clip=target_noise_clip,
                            policy_kwargs=policy_kwargs,
                            action_noise=action_noise,
                            verbose=1, tensorboard_log=tensorboard_log,
                            device="cuda", seed=args.seed)
                
            elif args.algo == "ddpg":
                model = DDPG(
                    "MlpPolicy", env,
                    learning_rate=float(params["learning_rate"]),
                    buffer_size=int(params["buffer_size"]),
                    batch_size=int(params["batch_size"]),
                    tau=float(params["tau"]),
                    gamma=float(params["gamma"]),
                    train_freq=int(params["train_freq"]),
                    gradient_steps=int(params["gradient_steps"]),
                    learning_starts=int(params["learning_starts"]),
                    policy_kwargs=policy_kwargs,
                    verbose=1, tensorboard_log=tensorboard_log,
                    device=args.device, seed=args.seed,
                )
            elif args.algo == "ppo":
                model = PPO(
                    "MlpPolicy", env,
                    learning_rate=float(params["learning_rate"]),
                    n_steps=int(params["n_steps"]),
                    batch_size=int(params["batch_size"]),
                    n_epochs=int(params["n_epochs"]),
                    gamma=float(params["gamma"]),
                    gae_lambda=float(params["gae_lambda"]),
                    clip_range=float(params["clip_range"]),
                    ent_coef=float(params["ent_coef"]),
                    vf_coef=float(params["vf_coef"]),
                    max_grad_norm=float(params["max_grad_norm"]),
                    policy_kwargs=policy_kwargs,
                    verbose=1, tensorboard_log=tensorboard_log,
                    device=args.device, seed=args.seed,
                )
            elif args.algo == "dqn":
                # Policy: build from JSON best params (n_layers x layer_size, activation)
                policy_kwargs = create_policy_kwargs(params, "dqn")

                # Use values EXACTLY as in your best_params JSON (trial 18)
                dqn_kwargs = dict(
                    learning_rate=float(params["learning_rate"]),                # 0.0001486...
                    buffer_size=int(params["buffer_size"]),                      # 72194
                    batch_size=int(params["batch_size"]),                        # 64
                    gamma=float(params["gamma"]),                                # 0.9969019...
                    train_freq=int(params["train_freq"]),                        # 4   <-- NOTE: int, not (k, "step")
                    learning_starts=int(params["learning_starts"]),              # 9552
                    target_update_interval=int(params["target_update_interval"]),# 1342
                    exploration_fraction=float(params["exploration_fraction"]),  # 0.0515408...
                    exploration_final_eps=float(params["exploration_final_eps"]),# 0.0741442...
                    policy_kwargs=policy_kwargs,
                    verbose=1, tensorboard_log=tensorboard_log,
                    device=args.device, seed=args.seed,
                )

                # Only include exploration_initial_eps if it exists in params (your best JSON doesn’t have it)
                if "exploration_initial_eps" in params:
                    dqn_kwargs["exploration_initial_eps"] = float(params["exploration_initial_eps"])

                model = DQN("MlpPolicy", env, **dqn_kwargs)
        # Train & save
        cb = EpisodeStatsLogger(log_path=log_file)
        print(f"Training {args.algo.upper()} for {args.timesteps} timesteps...")
        model.learn(total_timesteps=int(args.timesteps), callback=cb, log_interval=10)
        print("Training complete.")
        model.save(model_path)
        if args.algo in ["td3","sac","ddpg"]:
            try:
                model.save_replay_buffer(replay_buffer_path)
            except Exception as e:
                print(f"⚠️ Could not save replay buffer: {e}")
        env.save(os.path.join(model_base_dir, f"{args.algo}_vec_normalize_final.pkl"))
        env.close()

    if args.noise == "all":
        for _nl in [0, 0.001, 0.01, 0.1]:
            _train_one_noise(_nl)
        return
    else:
        # The old script had a bug where it would ignore --voltage-noise-std
        # when --noise=single. This ensures it's used.
        _train_one_noise(args.voltage_noise_std)
    # === END of change ===


if __name__ == "__main__":
    print("Starting training...")
    main()
