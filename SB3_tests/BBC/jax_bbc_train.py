# jax_bbc_train.py — Buck-Boost training (A2C, SAC, or TD3) with vectorized envs

import os
import argparse
import numpy as np
import torch.nn as nn
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
            avg_duty = float(ep.get("avg_duty", np.nan))  # NEW
            avg_voltage = float(ep.get("avg_voltage", np.nan))
            # CSV row
            self.log_file.write(f"{self.ep_idx},{total_r:.6f},{length},{avg_duty:.6f},{avg_voltage:.6f}\n")

            # Optional: raw values in TensorBoard
            if self.log_tensorboard:
                self.logger.record("custom/ep_reward_raw", total_r)
                self.logger.record("custom/ep_len_raw", length)
                self.logger.record("custom/ep_avg_duty_raw", avg_duty)  # NEW
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

def duty_bins_uniform(dmin=0.10, dmax=0.90, n=51):
    return np.linspace(dmin, dmax, n, dtype=np.float32)


def make_env(seed: int, rank: int = 0, k_macro: int = 1,
             algo_name: str = "td3",
             dqn_bins: int = 41,
             duty_min: float = 0.10,
             duty_max: float = 0.90):
    def _thunk():
        e = JAXBuckBoostConverterEnv(
            dt=5e-6,
            frame_skip=26,
            max_episode_steps=4000,
            grace_period_steps=200,
            target_voltage=-30.0,
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
    parser.add_argument("--algo", choices=["a2c", "sac", "td3", "dqn"], default="a2c")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel envs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--macro-k", type=int, default=1, help="Repeat each action for k PWM periods")
    # Arguments for randomizing input voltage
    parser.add_argument("--randomize-vin", action="store_true", help="Randomize input voltage each episode")
    parser.add_argument("--vin-min", type=float, default=46.5, help="Min input voltage for randomization")
    parser.add_argument("--vin-max", type=float, default=49.5, help="Max input voltage for randomization")

    parser.add_argument("--dqn-bins", type=int, default=41, help="Number of duty bins for DQN")
    parser.add_argument("--duty-min", type=float, default=0.10, help="Min duty for bins")
    parser.add_argument("--duty-max", type=float, default=0.90, help="Max duty for bins")

    args = parser.parse_args()

    algo_name = args.algo.upper()
    n_envs = max(1, args.n_envs)

    # Derived discount to keep physical horizon consistent when using macro steps
    gamma_phys_a2c = 0.995
    gamma_phys_sac = 0.9995
    gamma_phys_td3 = 0.99
    gamma_a2c = gamma_phys_a2c ** args.macro_k
    gamma_sac = gamma_phys_sac ** args.macro_k
    gamma_td3 = gamma_phys_td3 ** args.macro_k

    # base = trained_bbc_models/TD3
    base = os.path.join("trained_bbc_models", algo_name)
    os.makedirs(base, exist_ok=True)

    # unique run folder (e.g. TD3_15)
    run_name = f"{algo_name}_{args.seed}"   # or use trial number if tuning
    run_dir = os.path.join(base, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # log files inside run_dir
    log_file = os.path.join(run_dir, f"{args.algo}_training_log.txt")
    tensorboard_log = run_dir

    # ===== Vec envs =====
    env_fns = [make_env(args.seed, i, args.macro_k,
                        algo_name=args.algo,
                        dqn_bins=args.dqn_bins,
                        duty_min=args.duty_min,
                        duty_max=args.duty_max) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = vec_cls(env_fns)


    # Observation normalization is beneficial for all algorithms.
    # Reward normalization is typically used for off-policy algorithms but can sometimes hurt.
    # We are only normalizing observations here.
    if args.algo in ["sac", "td3"]:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    else: # A2C
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # ===== Policies =====
    model = None
    if args.algo == "a2c":
        policy_kwargs = dict(
            net_arch=dict(pi=[400, 400], vf=[400, 400]),
            activation_fn=nn.Tanh,
            ortho_init=False,
            log_std_init=-2.0,
        )
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=512,
            gamma=gamma_a2c,
            gae_lambda=0.90,
            ent_coef=0.005,
            vf_coef=1.0,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            use_rms_prop=True,
            rms_prop_eps=1e-5,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=args.device,
            seed=args.seed,
        )
    elif args.algo == "sac":
        policy_kwargs = dict(net_arch=[400, 400], activation_fn=nn.Tanh)
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=0.000105,
            buffer_size=200_000,
            batch_size=64,
            tau=0.00834,
            gamma=gamma_sac,
            train_freq=2,
            gradient_steps=16,
            learning_starts=10_000,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=args.device,
            seed=args.seed,
        )
    elif args.algo == "td3":
        act_dim = env.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(act_dim), sigma=0.06426859130004395 * np.ones(act_dim)) #this must be turned off for deterministic eval

        model = TD3(
            "MlpPolicy",
            env,                                    # VecNormalize'd SubprocVecEnv (8 envs) for parity
            learning_rate=2.8533260496524675e-04,
            buffer_size=800_000,
            batch_size=512,
            tau=0.014272704289474013,
            gamma=0.9760416404048181,
            train_freq=(1, "step"),
            gradient_steps=1,
            learning_starts=21958,                  # round from 0.0274468624 * 800000
            policy_delay=3,
            target_policy_noise=0.1464511120522956,
            target_noise_clip=0.3649340891310912,  # 2.4918492186 * policy_noise, capped to [0.1, 0.8]
            policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn=nn.ReLU),
            action_noise=action_noise,
            device="cuda",
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
        )

    elif args.algo == "dqn":
        # NOTE: Env should be wrapped with DiscretizedDutyWrapper using args.dqn_bins in [0.1, 0.9]
        # e.g., levels = np.linspace(0.10, 0.90, args.dqn_bins, dtype=np.float32)

        model = DQN(
            "MlpPolicy",
            env,                                    # VecNormalize'd SubprocVecEnv (8 envs) for parity
            learning_rate=2.0e-4,                   # near TD3’s LR scale
            buffer_size=800_000,                    # large replay works well with many bins
            batch_size=512,
            gamma=0.976,                            # mirrors TD3’s long horizon
            train_freq=(1, "step"),
            gradient_steps=1,
            learning_starts=22_000,                 # ~0.0275 * buffer_size (like your TD3)
            target_update_interval=3000,            # stabilize Q targets
            exploration_initial_eps=1.0,            # start fully exploratory
            exploration_final_eps=0.05,             # settle to modest exploration
            exploration_fraction=0.24,              # decay over ~24% of training
            policy_kwargs=dict(
                net_arch=[256, 256, 256],
                activation_fn=nn.ReLU,
                # dueling=True                        # helpful for control
            ),
            device="cuda",
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
        )




    cb = EpisodeStatsLogger(log_path=log_file)
    print(f"Training {algo_name} for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=cb,
        progress_bar=False,
        log_interval=10,
    )
    print("Training complete.")
    # ===== Save model + VecNormalize stats =====
    model.save(os.path.join(run_dir, f"{args.algo}_bbc_model_final"))
    env.save(os.path.join(run_dir, f"{args.algo}_vec_normalize_final.pkl"))

    env.close()

if __name__ == "__main__":
    print("Starting training...")
    main()
