#!/usr/bin/env python3
# bbc_jax_hp.py — Optuna-based hyperparameter tuning for the JAX Buck‑Boost Converter env
# Mirrors the style of ip_jax_hp.py, adapted for JAXBuckBoostConverterEnv.
# - Algorithms supported initially: SAC, A2C, DQN (DQN via duty discretization wrapper)
# - Ranges are factored in one place per algorithm to make later edits easy.
# - Includes optional macro-step wrapper (repeat action for k PWM periods) to speed up learning.
# - Saves best params and Optuna study DB for resuming.

import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

import optuna
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import A2C, SAC, DQN, PPO, TD3, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise  # <-- add this
from stable_baselines3.common.callbacks import BaseCallback


# Local env
from np_bbc_env import JAXBuckBoostConverterEnv


# ---------- Utilities ----------

def set_global_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

class HPTrialTBLogger(BaseCallback):
    """
    Per-episode TensorBoard logger for a single trial.
    Records raw training episode totals from Monitor:
      - custom/ep_reward_raw
      - custom/ep_len_raw
    """
    def __init__(self, trial_number: int):
        super().__init__()
        self.trial_number = trial_number
        self._ep_idx = 0

    def _on_training_start(self) -> None:
        # Tag the trial number once in the run
        self.logger.record("trial/number", int(self.trial_number))

    def _on_step(self) -> bool:
        wrote = False
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            self._ep_idx += 1
            # Raw, unsmoothed values straight from Monitor
            self.logger.record("custom/ep_reward_raw", float(ep.get("r", 0.0)))
            self.logger.record("custom/ep_len_raw", int(ep.get("l", 0)))
            self.logger.record("custom/ep_avg_duty", float(info["avg_duty"]))
            self.logger.record("custom/ep_avg_voltage", float(info["avg_voltage"]))
            # Optional: index, if you want it in TB
            self.logger.record("custom/episode_idx", int(self._ep_idx))
            wrote = True
        # Flush immediately when an episode finishes so points show up right away
        if wrote:
            self.logger.dump(step=int(self.num_timesteps))
        return True

# Macro-step wrapper (repeat action for k PWM periods), adapted to gymnasium.
class MultiPeriodStep(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int = 1):
        super().__init__(env)
        assert k >= 1
        self.k = int(k)

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


# Discretize a 1-D continuous duty command into N fixed levels (for DQN).
class DiscretizedDutyWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, duty_levels: Sequence[float]):
        super().__init__(env)
        self.duty_levels = np.asarray(duty_levels, dtype=np.float32)
        assert self.duty_levels.ndim == 1 and len(self.duty_levels) >= 2
        self.action_space = spaces.Discrete(len(self.duty_levels))

    def action(self, act: int):
        d = float(self.duty_levels[int(act)])
        return np.array([d], dtype=np.float32)


@dataclass
class TuneConfig:
    total_timesteps: int = 150_000          # small by default; change later
    eval_interval: int = 30_000
    n_eval_episodes: int = 5
    tb_root: str = "./bbc_41_hp_logs"
    results_dir: str = "./bbc_41_hp_results"
    storage_tpl: str = "sqlite:///bbc_41_jax_optuna_{algo}.db"


def make_envs(seed: int, algo_name: str, macro_k: int = 1, dqn_bins: int = 41,
              n_envs: int = 8, duty_min: float = 0.1, duty_max: float = 0.9):
    def _thunk(rank: int):
        def _make():
            e = JAXBuckBoostConverterEnv(
                dt=5e-6,
                frame_skip=26,
                max_episode_steps=4000,
                grace_period_steps=200,        # align with training
                target_voltage=-30.0,
            )
            # match training: wider reward clip if env exposes it
            if hasattr(e, "_clip_low"):  e._clip_low  = -10.0
            if hasattr(e, "_clip_high"): e._clip_high =  10.0   # :contentReference[oaicite:1]{index=1}

            if macro_k > 1:
                e = MultiPeriodStep(e, k=macro_k)

            if algo_name.lower() == "dqn":
                levels = np.linspace(duty_min, duty_max, num=int(dqn_bins), dtype=np.float32)
                e = DiscretizedDutyWrapper(e, duty_levels=levels)

            # add avg_duty / avg_voltage like training
            e = Monitor(e, info_keywords=("iL", "vC", "mag_vC", "e_norm", "dduty", "in_band",
                                          "avg_duty", "avg_voltage"))  # :contentReference[oaicite:2]{index=2}
            e.reset(seed=seed + rank)
            return e
        return _make

    env_fns = [_thunk(i) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv  # :contentReference[oaicite:3]{index=3}
    return vec_cls(env_fns)



# ---------- Algorithm-specific hyperparam spaces (edit later) ----------

def activation_from_name(name: str):
    return {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU}[name]


def suggest_policy_kwargs(trial: optuna.Trial):
    n_layers = trial.suggest_categorical("n_layers", [1, 3])  # shallow wins here
    layer_size = trial.suggest_int("layer_size", 64, 512, log=True)
    act_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu", "elu"])
    return {
        "net_arch": [layer_size] * n_layers,                 # straight architecture
        "activation_fn": activation_from_name(act_name)
    }

def suggest_action_noise_sigma(trial: optuna.Trial):
    # Rollout exploration noise for continuous actions
    return trial.suggest_float("action_noise_sigma", 0.03, 0.30, log=True)



def suggest_a2c(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 3e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
        "n_steps": trial.suggest_int("n_steps", 32, 2048, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "ent_coef": trial.suggest_float("ent_coef", 1e-7, 0.05, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
        "rms_prop_eps": trial.suggest_float("rms_prop_eps", 1e-6, 1e-3, log=True),
        "use_rms_prop": trial.suggest_categorical("use_rms_prop", [True, False]),
    }


def suggest_sac(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 100_000, 400_000, step=50_000),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=32),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", 0.001, 0.01, 0.1]),
        "train_freq": trial.suggest_categorical("train_freq", [1, 2, 4]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4, 8, 16]),
        "learning_starts": trial.suggest_int("learning_starts", 5_000, 20_000, step=2_500),
    }



def suggest_dqn(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 50_000, 200_000),
        "learning_starts": trial.suggest_int("learning_starts", 5000, 20000),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.2),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32]),
        "target_update_interval": trial.suggest_int("target_update_interval", 500, 2000),
    }



def suggest_ppo(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "n_steps": trial.suggest_int("n_steps", 64, 4096, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024]),
        "n_epochs": trial.suggest_int("n_epochs", 5, 20),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-7, 0.02, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
    }


def suggest_td3(trial: optuna.Trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)

    buffer_size = trial.suggest_int("buffer_size", 200_000, 1_400_000, step=200_000)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])  # larger often better

    tau = trial.suggest_float("tau", 0.003, 0.02)       # SB3 default 0.005 is strong
    gamma = trial.suggest_float("gamma", 0.96, 0.995)   # narrower = easier search

    train_freq_k = trial.suggest_categorical("train_freq_k", [1])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1])

    ls_frac = trial.suggest_float("learning_starts_frac", 0.01, 0.10)
    learning_starts = int(max(5_000, ls_frac * buffer_size))

    policy_delay = trial.suggest_categorical("policy_delay", [1, 2, 3])

    policy_noise = trial.suggest_float("target_policy_noise", 0.05, 0.40)
    clip_mult = trial.suggest_float("target_noise_clip_mult", 1.0, 3.0)
    target_noise_clip = min(0.8, max(0.1, clip_mult * policy_noise))  # ensure clip >= noise

    return {
        "learning_rate": lr,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "tau": tau,
        "gamma": gamma,
        "train_freq": (train_freq_k, "step"),
        "gradient_steps": gradient_steps,
        "learning_starts": learning_starts,
        "policy_delay": policy_delay,
        "target_policy_noise": policy_noise,
        "target_noise_clip": target_noise_clip,
        "action_noise_sigma": suggest_action_noise_sigma(trial),
    }



def suggest_ddpg(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 100_000, 500_000, step=50_000),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=32),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
        "train_freq": trial.suggest_categorical("train_freq", [1, 2, 4]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4, 8, 16]),
        "learning_starts": trial.suggest_int("learning_starts", 5_000, 30_000, step=2_500),
    }



def build_model(algo: str, env, trial: optuna.Trial, device: str, cfg):
    algo = algo.lower()
    pk = suggest_policy_kwargs(trial)

    if algo == "a2c":
        params = suggest_a2c(trial)
        return A2C("MlpPolicy", env, device=device, verbose=0,
                   tensorboard_log=os.path.join(cfg.tb_root, algo),
                   policy_kwargs=pk, **params)

    if algo == "sac":
        params = suggest_sac(trial)
        # Smaller nets are often enough for 1‑D action
        if "net_arch" not in pk: pk["net_arch"] = [256, 256]
        return SAC("MlpPolicy", env, device=device, verbose=0,
                   tensorboard_log=os.path.join(cfg.tb_root, algo),
                   policy_kwargs=pk, **params)

    if algo == "dqn":
        params = suggest_dqn(trial)
        return DQN("MlpPolicy", env, device=device, verbose=0,
                   tensorboard_log=os.path.join(cfg.tb_root, algo),
                   policy_kwargs=pk, **params)


    if algo == "ppo":
        params = suggest_ppo(trial)
        return PPO("MlpPolicy", env, device=device, verbose=0,
                   tensorboard_log=os.path.join(cfg.tb_root, algo),
                   policy_kwargs=pk, **params)

    if algo == "td3":
        params = suggest_td3(trial)
        # if "net_arch" not in pk: pk["net_arch"] = [256, 256]

        # Build action noise from suggested sigma
        sigma = params.pop("action_noise_sigma")  # remove from kwargs
        act_dim = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(act_dim),
            sigma=sigma * np.ones(act_dim)
        )
        print(f"\n===== Trial {trial.number} ({algo}) =====")
        print("Params:", json.dumps(params, indent=2))
        print("Policy kwargs:", json.dumps(
            {k: str(v) for k, v in pk.items()}, indent=2
        ))

        return TD3(
            "MlpPolicy", env, device=device, verbose=0,
            tensorboard_log=os.path.join(cfg.tb_root, algo),
            policy_kwargs=pk,
            action_noise=action_noise,   # <--- wired in
            **params
        )


    if algo == "ddpg":
        params = suggest_ddpg(trial)
        if "net_arch" not in pk: pk["net_arch"] = [256, 256]
        return DDPG("MlpPolicy", env, device=device, verbose=0,
                    tensorboard_log=os.path.join(cfg.tb_root, algo),
                    policy_kwargs=pk, **params)
    
    

    raise ValueError(f"Unsupported algo: {algo}")
    



def objective(trial: optuna.Trial, algo: str, seed: int, macro_k: int, dqn_bins: int, device: str, cfg):
    set_global_seeds(seed)

    # vectorized envs
    env_train = make_envs(seed=seed, algo_name=algo, macro_k=macro_k, dqn_bins=dqn_bins, n_envs=args.n_envs)
    env_train = VecNormalize(env_train, norm_obs=True, norm_reward=False)  # same as training :contentReference[oaicite:5]{index=5}

    # separate single-env eval, share obs stats
    env_eval  = make_envs(seed=seed + 100, algo_name=algo, macro_k=macro_k, dqn_bins=dqn_bins, n_envs=1)
    env_eval  = VecNormalize(env_eval, norm_obs=True, norm_reward=False)
    env_eval.training = False
    env_eval.norm_reward = False
    env_eval.obs_rms = env_train.obs_rms  # sync stats for fair evaluation


    model = build_model(algo, env_train, trial, device=device, cfg=cfg)
    

    # NEW: training per-episode TB logger
    tb_ep_logger = HPTrialTBLogger(trial.number)

    timesteps = 0
    best_metric = -np.inf
    eval_idx = 0

    while timesteps < cfg.total_timesteps:
        model.learn(
            cfg.eval_interval,
            reset_num_timesteps=False,
            progress_bar=False,
            tb_log_name=f"T{trial.number}",
            callback=tb_ep_logger,           # <-- attach TB logger
        )
        timesteps += cfg.eval_interval
        eval_idx += 1

        # Evaluate and log raw eval episodes to TensorBoard
        ep_rewards, ep_lengths = evaluate_policy(
            model, env_eval, n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True, return_episode_rewards=True
        )

        # --- TensorBoard logging for evaluation ---
        # Per-episode raw values (one scalar per episode at the same 'step')
        for i, (r, L) in enumerate(zip(ep_rewards, ep_lengths), start=1):
            model.logger.record("eval/ep_reward_raw", float(r))
            model.logger.record("eval/ep_len_raw", int(L))
            model.logger.record("eval/episode_idx", int(i))
        # Summaries
        r_arr = np.asarray(ep_rewards, dtype=float)
        L_arr = np.asarray(ep_lengths, dtype=int)
        top_k = int(max(1, np.ceil(0.6 * len(r_arr))))
        topk_avg = float(np.mean(np.sort(r_arr)[-top_k:]))

        model.logger.record("eval/mean_reward", float(r_arr.mean()))
        model.logger.record("eval/std_reward", float(r_arr.std(ddof=0)))
        model.logger.record("eval/mean_len", float(L_arr.mean()))
        model.logger.record("eval/topk_avg_reward", topk_avg)
        model.logger.record("eval/chunk_index", int(eval_idx))
        model.logger.dump(step=int(timesteps))   # flush eval metrics now

        # # --- HARD PRUNE: at 500k steps if metric not above 200 ---
        # if timesteps >= 500_000 and topk_avg <= 1000.0:
        #     model.logger.record("prune/hard_pruned", 1)
        #     model.logger.record("prune/hard_prune_at", int(timesteps))
        #     model.logger.record("prune/topk_avg_at_prune", float(topk_avg))
        #     model.logger.dump(step=int(timesteps))
        #     env_train.close(); env_eval.close()
        #     raise optuna.TrialPruned(f"Hard-pruned at {timesteps} steps: topk_avg={topk_avg:.3f} ≤ 200.")
        
        # # --- HARD PRUNE: at 500k steps if metric not above 200 ---
        # if timesteps >= 200_000 and topk_avg <= -900.0:
        #     model.logger.record("prune/hard_pruned", 1)
        #     model.logger.record("prune/hard_prune_at", int(timesteps))
        #     model.logger.record("prune/topk_avg_at_prune", float(topk_avg))
        #     model.logger.dump(step=int(timesteps))
        #     env_train.close(); env_eval.close()
        #     raise optuna.TrialPruned(f"Hard-pruned at {timesteps} steps: topk_avg={topk_avg:.3f} < -1000.")

        # Optuna bookkeeping / pruning
        trial.report(topk_avg, step=timesteps)
        if trial.should_prune():
            env_train.close(); env_eval.close()
            raise optuna.TrialPruned()

        best_metric = max(best_metric, topk_avg)

    env_train.close()
    env_eval.close()
    return best_metric




def optimize(algo: str, n_trials: int, n_jobs: int, seed: int, macro_k: int, dqn_bins: int, device: str, cfg):
    os.makedirs(cfg.tb_root, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    storage = cfg.storage_tpl.format(algo=algo.lower())
    study = optuna.create_study(
        direction="maximize",
        study_name=f"bbc_{algo.lower()}_tuning",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=cfg.eval_interval,  # use one eval chunk as the minimum
            reduction_factor=2,
            min_early_stopping_rate=0,
        ),
    )

    pbar = tqdm(total=n_trials, desc=f"Tuning {algo.upper()}", dynamic_ncols=True)
    def _cb(st, tr):
        pbar.update(1)
        try:
            pbar.set_postfix(best_val=f"{st.best_value:.2f}")
        except Exception:
            pbar.set_postfix(best_val="–")

    study.optimize(
        lambda t: objective(t, algo, seed, macro_k, dqn_bins, device, cfg),
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[_cb],
        show_progress_bar=False,
        gc_after_trial=True,
    )
    pbar.close()

    out_json = os.path.join(cfg.results_dir, f"{algo.lower()}_best_params.json")
    with open(out_json, "w") as f:
        json.dump({
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": n_trials
        }, f, indent=2)

    print(f"[{algo.upper()}] Best reward: {study.best_value:.2f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["sac", "a2c", "dqn", "ppo", "td3", "ddpg"], default="sac")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--n-parallel", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--macro-k", type=int, default=1, help="Repeat each action for k PWM periods")
    ap.add_argument("--dqn-bins", type=int, default=41, help="Number of duty bins for DQN (spaced in [duty_min, duty_max])")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    # Quick tweaks without editing code
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--eval-interval", type=int, default=100_000)
    ap.add_argument("--n-eval-episodes", type=int, default=1)
    ap.add_argument("--n-envs", type=int, default=8, help="Parallel envs")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Global (editable) config
    cfg = TuneConfig()
    if args.total_timesteps is not None:
        cfg.total_timesteps = int(args.total_timesteps)
    if args.eval_interval is not None:
        cfg.eval_interval = int(args.eval_interval)
    if args.n_eval_episodes is not None:
        cfg.n_eval_episodes = int(args.n_eval_episodes)

    print(f"Tuning {args.algo.upper()} on JAXBuckBoostConverterEnv...")
    print(f"Device: {args.device}, seed: {args.seed}, macro_k: {args.macro_k}")

    optimize(
        algo=args.algo,
        n_trials=args.n_trials,
        n_jobs=args.n_parallel,
        seed=args.seed,
        macro_k=args.macro_k,
        dqn_bins=41,
        device=args.device,
        cfg=cfg,
    )
 
