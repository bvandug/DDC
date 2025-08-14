# jax_bbc_train.py — Buck-Boost training (A2C or SAC) with vectorized envs + fast SAC options

import os
import argparse
import numpy as np
import torch.nn as nn

from stable_baselines3 import A2C, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from np_bbc_env import JAXBuckBoostConverterEnv


# ===== Logging callback (works with VecEnvs) =====
class EpisodeStatsLogger(BaseCallback):
    def __init__(self, log_path, **kwargs):
        super().__init__(**kwargs)
        self.log_path = log_path
        self.log_file = None
        self.ep_idx = 0

    def _on_training_start(self) -> None:
        self.log_file = open(self.log_path, "w")
        self.log_file.write("Episode,Total Reward,Length\n")

    def _on_step(self) -> bool:
        # Monitor injects {"episode": {"r": return, "l": length}} when an env finishes
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.ep_idx += 1
                self.log_file.write(f"{self.ep_idx},{ep['r']:.4f},{ep['l']}\n")
        return True

    def _on_training_end(self) -> None:
        self.log_file.write("Training completed.\n")
        self.log_file.close()


# ===== Macro-step wrapper: repeat the same action for k PWM periods =====
import gymnasium as gym

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


def make_env(seed: int, rank: int = 0, k_macro: int = 1):
    def _thunk():
        e = JAXBuckBoostConverterEnv(
            dt=5e-6,
            frame_skip=10,          # 20 kHz
            max_episode_steps=4000, # 0.2 s/episode at k_macro=1
            grace_period_steps=100,
            target_voltage=-30.0,
            enforce_dcm=True,
        )
        # For SAC + reward normalization, wider per-step clip helps (no double squashing)
        if hasattr(e, "_clip_low"):  e._clip_low  = -10.0
        if hasattr(e, "_clip_high"): e._clip_high =  10.0

        # Macro-step (repeat action k times)
        if k_macro > 1:
            e = MultiPeriodStep(e, k=k_macro)

        e = Monitor(
            e,
            info_keywords=("iL", "vC", "mag_vC", "e_norm", "dduty", "in_band"),
        )
        e.reset(seed=seed + rank)
        return e
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["a2c", "sac"], default="a2c")
    parser.add_argument("--timesteps", type=int, default=2_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel envs (used for both A2C and SAC)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--macro-k", type=int, default=1, help="Repeat each action for k PWM periods (speeds up training)")
    args = parser.parse_args()

    algo_name = args.algo.upper()
    n_envs = max(1, args.n_envs)

    # Derived discount to keep physical horizon consistent when using macro steps
    # Per-PWM-period discount (50 µs at 20 kHz)
    gamma_phys_a2c = 0.995     # was 0.995 per step
    gamma_phys_sac = 0.9995    # longer horizon for SAC
    gamma_a2c = gamma_phys_a2c ** args.macro_k
    gamma_sac = gamma_phys_sac ** args.macro_k

    # Paths
    base = os.path.join("trained_bbc_models", algo_name)
    os.makedirs(base, exist_ok=True)
    log_file = os.path.join(base, f"{args.algo}_training_log.txt")
    tensorboard_log = base

    # ===== Vec envs =====
    # Use subprocesses for speed (works for both A2C & SAC).
    env_fns = [make_env(args.seed, i, args.macro_k) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = vec_cls(env_fns)

    # Reward normalization only for SAC (off-policy Q stability).
    if args.algo == "sac":
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # ===== Policies =====
    if args.algo == "a2c":
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
            ortho_init=False,
            log_std_init=-2.0,
        )
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,                 # 256 * n_envs samples/update
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
    else:
        # SAC: fast & stable profile
        policy_kwargs = dict(net_arch=[128, 128], activation_fn=nn.Tanh)  # smaller net = faster
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=gamma_sac,            # adjust for macro-k
            train_freq=1,
            gradient_steps=1,
            learning_starts=30_000 if args.macro_k == 1 else 15_000,
            ent_coef="auto_0.2",        # sensible initial alpha
            target_entropy=-0.3,        # lower target for 1-D action
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=args.device,
            seed=args.seed,
        )

    # ===== Train =====
    cb = EpisodeStatsLogger(log_path=log_file)
    model.learn(
        total_timesteps=args.timesteps,
        callback=cb,
        progress_bar=False,
        log_interval=10,
    )

    # ===== Save model + VecNormalize stats =====
    model.save(os.path.join(base, f"{args.algo}_bbc_model_final"))
    env.save(os.path.join(base, f"{args.algo}_vec_normalize_final.pkl"))
    env.close()


if __name__ == "__main__":
    main()
