#!/usr/bin/env python3
"""
sac_vec_consistency_test.py
---------------------------
Smoke test to verify SAC learns similarly with 1 env vs 8 envs, when
hyperparameters are scaled properly (esp. gradient_steps ~= n_envs to keep UTD≈1).

It trains two runs back-to-back:
  - Run A: n_envs=1
  - Run B: n_envs=8 (SubprocVecEnv)

Both use:
  • JAXBuckBoostConverterEnv (20 kHz: dt=5e-6, frame_skip=10; 1 PWM per RL step)
  • Reward normalization (VecNormalize norm_reward=True, clip_reward=10.0)
  • SAC tuned for fine time steps (gamma=0.9995, ent_coef='auto_0.2', target_entropy=-0.3)
  • UTD matched: train_freq=1, gradient_steps=n_envs

At the end, we evaluate both models deterministically for N episodes
and print a side-by-side table and save results to CSV.

Usage (from the repo root):
  python sac_vec_consistency_test.py --timesteps 200000 --seed 42 --eval-episodes 10
"""

import os
import argparse
import numpy as np
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from np_bbc_env import JAXBuckBoostConverterEnv


def make_env(seed: int, rank: int = 0):
    def _thunk():
        e = JAXBuckBoostConverterEnv(
            dt=5e-6,
            frame_skip=10,           # 20 kHz, 1 PWM per step
            max_episode_steps=4000,  # 0.2 s episodes
            grace_period_steps=100,
            target_voltage=-30.0,
            enforce_dcm=True,
        )
        # widen per-step clip to play well with reward normalization
        if hasattr(e, "_clip_low"):  e._clip_low  = -10.0
        if hasattr(e, "_clip_high"): e._clip_high =  10.0
        e = Monitor(e)
        e.reset(seed=seed + rank)
        return e
    return _thunk


def train_sac(n_envs: int, total_timesteps: int, seed: int, logdir: str):
    # Build vec env
    env_fns = [make_env(seed, i) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = vec_cls(env_fns)
    # Normalize both obs and rewards for stable Q targets
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # Keep UTD ≈ 1 regardless of n_envs
    utd = n_envs  # gradient steps per vector step
    policy_kwargs = dict(net_arch=[128, 128], activation_fn=nn.Tanh)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.9995,              # per PWM step
        train_freq=1,              # every vector step
        gradient_steps=utd,        # ≈1 grad step per env transition
        learning_starts=30_000,
        ent_coef="auto_0.2",
        target_entropy=-0.3,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=logdir,
        device="auto",
        seed=seed,
    )

    run_name = f"SAC_{n_envs}env"
    print(f"\n=== Training {run_name} for {total_timesteps} steps ===")
    model.learn(total_timesteps=total_timesteps, log_interval=10, progress_bar=False)

    # Save model + VecNormalize stats
    os.makedirs(logdir, exist_ok=True)
    model_path = os.path.join(logdir, f"{run_name}_model")
    norm_path  = os.path.join(logdir, f"{run_name}_vecnorm.pkl")
    model.save(model_path)
    env.save(norm_path)

    # Build a fresh eval env and load normalization stats
    eval_env = DummyVecEnv([make_env(seed+1234, 0)])
    eval_env = VecNormalize.load(norm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=False)
    print(f"{run_name} eval: mean_reward={mean_r:.3f} ± {std_r:.3f}")

    # Close
    env.close()
    eval_env.close()

    return dict(run=run_name, n_envs=n_envs, steps=total_timesteps, mean_reward=mean_r, std_reward=std_r,
                model_path=model_path, vecnorm_path=norm_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--logdir", default="trained_bbc_models/consistency_smoke")
    args = p.parse_args()

    results = []
    for n_envs in [1, 8]:
        res = train_sac(n_envs=n_envs, total_timesteps=args.timesteps, seed=args.seed, logdir=args.logdir)
        results.append(res)

    # Print side-by-side
    print("\n=== Summary (deterministic eval) ===")
    for r in results:
        print(f"{r['run']}: mean_reward={r['mean_reward']:.3f} ± {r['std_reward']:.3f}  (n_envs={r['n_envs']}, steps={r['steps']})")

    # Save CSV
    import csv
    csv_path = os.path.join(args.logdir, "sac_vec_consistency_results.csv")
    os.makedirs(args.logdir, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run","n_envs","steps","mean_reward","std_reward","model_path","vecnorm_path"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
