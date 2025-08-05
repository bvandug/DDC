import os
import optuna
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from nic_bbc_env import BuckBoostConverterEnv  # Adjust this import if needed
from stable_baselines3.common.vec_env import VecNormalize
# ───── Global Constants ─────────────────────
TOTAL_TIMESTEPS = 200_000
EVAL_EPISODES = 5
N_TRIALS = 30
SEED = 42

# ───── Objective Function for Optuna ─────────
def objective(trial):
    # ── Sample hyperparameters ───────────────
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.90, 0.999)
    n_steps = trial.suggest_categorical('n_steps', [8, 16, 32, 64, 128])
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 1e-2, log=True)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)

    # ── Create environment ───────────────────
    def make_env():
        env = BuckBoostConverterEnv()
        env = Monitor(env)
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── Define A2C model ─────────────────────
    model = A2C(
        "MlpPolicy",
        env,
        verbose=0,
        seed=SEED,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        device="auto"
    )

    # ── Train ────────────────────────────────
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # ── Evaluate ─────────────────────────────
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES, return_episode_rewards=False)

    env.close()
    return mean_reward

# ───── Run Optuna Study ──────────────────────
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="a2c-bbc-tuning")
    study.optimize(objective, n_trials=N_TRIALS)

    print("🔎 Best Trial:")
    trial = study.best_trial
    print(f"  Mean Reward: {trial.value}")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
