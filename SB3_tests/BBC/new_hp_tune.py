import os
import json
import math
import optuna
import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Any, Optional, Tuple, Callable, List

import gymnasium as gym
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# --- ENV: your NumPy BBC env ---
from np_bbc_env import JAXBuckBoostConverterEnv  # action: continuous duty in [0.1, 0.9]

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


def make_env(seed: int, rank: int = 0, k_macro: int = 1) -> Callable[[], gym.Env]:
    """
    Factory for (vectorized) BBC envs. Mirrors your training script’s defaults.
    """
    def _thunk():
        env = JAXBuckBoostConverterEnv(
            dt=5e-6,
            frame_skip=26,            # 20 kHz switching
            max_episode_steps=4000,   # ~0.2 s per episode at k_macro=1
            grace_period_steps=100,
            target_voltage=-30.0,
        )

        # Wider per-step reward clip helps SAC (matches your training script tweak)
        if hasattr(env, "_clip_low"):  env._clip_low  = -10.0
        if hasattr(env, "_clip_high"): env._clip_high =  10.0

        if k_macro > 1:
            env = MultiPeriodStep(env, k=k_macro)

        env = Monitor(env, info_keywords=("iL", "vC", "err"))
        # Gymnasium reset returns (obs, info)
        obs, _ = env.reset(seed=seed + rank)
        return env
    return _thunk


class HyperparamTunerBBC:
    """
    Optuna tuner for the Buck-Boost env using SB3 (A2C / SAC).
    - Supports parallel VecEnvs (SubprocVecEnv).
    - Supports macro-step K (repeat action for K PWM periods).
    - Applies physical-horizon-aware gamma adjustment when macro-stepping.
    - Uses VecNormalize(normalize obs, do not normalize reward) by default.
    """

    def __init__(
        self,
        algo: str = "a2c",
        n_envs: int = 8,
        macro_k: int = 1,
        total_timesteps: int = 300_000,
        eval_interval: int = 25_000,
        n_eval_episodes: int = 8,
        study_name: Optional[str] = None,
        storage_url: Optional[str] = None,
        tensorboard_root: str = "bbc_hp_tb",
        seed: int = 42,
        device: str = "auto",
    ):
        self.algo = algo.lower()
        assert self.algo in {"a2c", "sac"}, "BBC env uses continuous action; supported algos: a2c, sac."

        self.n_envs = max(1, int(n_envs))
        self.macro_k = max(1, int(macro_k))
        self.total_timesteps = int(total_timesteps)
        self.eval_interval = int(eval_interval)
        self.n_eval_episodes = int(n_eval_episodes)
        self.study_name = study_name or f"bbc_{self.algo}_tuning"
        self.storage_url = storage_url or f"sqlite:///{self.study_name}.db"
        self.tensorboard_root = tensorboard_root
        self.seed = int(seed)
        self.device = device

        # Physical per-PWM-period discounts (mirrors your trainer choices)
        self.gamma_phys_a2c = 0.995
        self.gamma_phys_sac = 0.9995

        os.makedirs(self.tensorboard_root, exist_ok=True)
        os.makedirs("bbc_hp_results", exist_ok=True)

    # ---------- VecEnv construction ----------
    def _make_vec_env(self) -> VecNormalize:
        env_fns = [make_env(self.seed, i, self.macro_k) for i in range(self.n_envs)]
        VecCls = SubprocVecEnv if self.n_envs > 1 else DummyVecEnv
        vec = VecCls(env_fns)
        # Normalize obs; leave reward raw (as in your trainer)
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
        return vec

    # --- Replace this whole function ---
    def _suggest_policy_kwargs(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Use a flat, shared MLP for both actor & critic with identical widths/layers.
        Works for A2C and SAC. Keeps the search small & robust.
        """
        activation_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
        activation = trial.suggest_categorical("activation_fn", list(activation_map.keys()))
        width      = trial.suggest_int("width", 64, 512, step=64)
        layers     = trial.suggest_int("layers", 1, 3)

        # Shared body for both actor & critic
        return dict(
            net_arch=[width] * layers,          # flat shared MLP
            activation_fn=activation_map[activation],
            ortho_init=False
        )


    # ---------- Algo params ----------
    def _algo_and_params(self, trial: optuna.Trial, env: VecNormalize):
        # --- In _algo_and_params(), replace ONLY the A2C branch with this ---
        if self.algo == "a2c":
            gamma = self.gamma_phys_a2c ** self.macro_k
            policy_kwargs = self._suggest_policy_kwargs(trial)

            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "n_steps": trial.suggest_categorical("n_steps", [128, 256, 512, 1024]),
                "gamma": gamma,
                "gae_lambda": trial.suggest_float("gae_lambda", 0.85, 0.98),
                "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True),
                "vf_coef": trial.suggest_float("vf_coef", 0.5, 1.0),
                "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
                "use_rms_prop": True,
                "rms_prop_eps": 1e-5,
                "device": self.device,
                "seed": self.seed,
                "verbose": 0,
                "tensorboard_log": self.tensorboard_root,
                "policy_kwargs": policy_kwargs,
            }
            model = A2C("MlpPolicy", env, **params)
            return model


        else:  # SAC
            gamma = self.gamma_phys_sac ** self.macro_k

            policy_kwargs = self._suggest_policy_kwargs(trial)
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True),
                "buffer_size": trial.suggest_int("buffer_size", 200_000, 1_000_000, step=200_000),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "tau": trial.suggest_float("tau", 0.002, 0.02, log=True),
                "gamma": gamma,
                "train_freq": 1,
                "gradient_steps": 1,
                "learning_starts": trial.suggest_int("learning_starts", 10_000, 40_000, step=5_000),
                "ent_coef": trial.suggest_categorical("ent_coef", ["auto_0.1", "auto_0.2", "auto_0.5", "auto"]),
                "target_entropy": trial.suggest_categorical("target_entropy", [-0.3, -0.5, -1.0]),
                "device": self.device,
                "seed": self.seed,
                "verbose": 0,
                "tensorboard_log": self.tensorboard_root,
                "policy_kwargs": policy_kwargs,
            }
            model = SAC("MlpPolicy", env, **params)
            return model

    # ---------- Objective ----------
    def _objective(self, trial: optuna.Trial) -> float:
        # Construct a fresh vec env per trial (keeps VecNormalize stats per-trial)
        env = self._make_vec_env()
        model = self._algo_and_params(trial, env)

        timesteps = 0
        best_score = -float("inf")

        # Train/eval schedule
        while timesteps < self.total_timesteps:
            model.learn(self.eval_interval, reset_num_timesteps=False, progress_bar=False, tb_log_name=f"T{trial.number}")
            timesteps += self.eval_interval

            # Evaluate (deterministic) on the same normalized VecEnv
            mean_reward, std_reward = evaluate_policy(
                model,
                env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=False,
                warn=False,
            )

            # Track best
            best_score = max(best_score, float(mean_reward))

            # Report to Optuna (for pruners)
            trial.report(float(mean_reward), step=timesteps)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Save VecNormalize (optional): not necessary for tuner’s output, but handy for reproducing evaluation
        try:
            save_name = f"{self.algo}_vecnormalize_trial{trial.number}.pkl"
            env.save(os.path.join("bbc_hp_results", save_name))
        except Exception:
            pass

        # Clean up subprocesses
        env.close()
        return float(best_score)

    # ---------- Public API ----------
    def tune(self, n_trials: int = 40, n_jobs: int = 4) -> Dict[str, Any]:
        """
        Run Optuna with SuccessiveHalving pruner and return best params.
        """
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=max(3 * self.eval_interval, self.eval_interval),  # steps, not episodes; coarse guard
            reduction_factor=2,
            min_early_stopping_rate=0,
        )

        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            study_name=self.study_name,
            storage=self.storage_url,
            load_if_exists=True,
        )

        study.optimize(self._objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

        # Persist best
        out = {
            "algo": self.algo,
            "macro_k": self.macro_k,
            "n_envs": self.n_envs,
            "total_timesteps": self.total_timesteps,
            "eval_interval": self.eval_interval,
            "n_eval_episodes": self.n_eval_episodes,
            "best_value": study.best_value,
            "best_params": study.best_params,
        }
        with open(os.path.join("bbc_hp_results", f"{self.algo}_best_params.json"), "w") as f:
            json.dump(out, f, indent=2)

        print(f"[BBC {self.algo.upper()}] Best reward: {study.best_value:.3f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        return out


# ---------- CLI helper ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["a2c", "sac"], default="a2c")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--macro-k", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--eval-interval", type=int, default=25_000)
    parser.add_argument("--n-eval-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    tuner = HyperparamTunerBBC(
        algo=args.algo,
        n_envs=args.n_envs,
        macro_k=args.macro_k,
        total_timesteps=args.total_timesteps,
        eval_interval=args.eval_interval,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed,
        device=args.device,
        tensorboard_root=f"bbc_hp_tb/{args.algo}",
        study_name=f"bbc_{args.algo}_tuning",
        storage_url=f"sqlite:///bbc_optuna_{args.algo}.db",
    )
    tuner.tune(n_trials=args.n_trials, n_jobs=args.n_jobs)
