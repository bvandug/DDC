import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import optuna
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# -----------------------------
# Settings
# -----------------------------
NUM_ENVS      = 8
TOTAL_STEPS   = 200_000
EVAL_EPISODES = 5
BASE_SEED     = 42
DRONE_MODEL   = DroneModel.CF2X
PHYSICS       = Physics.PYB
USE_GUI       = False

# Activation mapping
activation_map = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU
}

# -----------------------------
# Environment factory
# -----------------------------
def make_env(rank: int):
    def _init():
        env = HoverAviary(drone_model=DRONE_MODEL,
                          physics=PHYSICS,
                          gui=USE_GUI)
        env = Monitor(env)
        seed = BASE_SEED + rank
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        return env
    return _init

# -----------------------------
# Policy kwargs helper
# -----------------------------
def policy_kwargs_from_trial(trial: optuna.Trial) -> dict:
    layer_size = trial.suggest_int('layer_size', 32, 512, log=True)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    arch = [layer_size] * n_layers
    act_choice = trial.suggest_categorical('activation_fn', list(activation_map.keys()))
    activation_fn = activation_map[act_choice]
    return {
        'net_arch': dict(pi=arch, vf=arch),
        'activation_fn': activation_fn
    }

# -----------------------------
# Objective for Optuna
# -----------------------------
def objective(trial: optuna.Trial) -> float:
    # Architecture
    policy_kwargs = policy_kwargs_from_trial(trial)
    # A2C hyperparameters
    params = {
        'n_steps': trial.suggest_int('n_steps', 16, 256, log=True),
        'gamma': trial.suggest_float('gamma', 0.90, 0.9999),
        'ent_coef': trial.suggest_float('ent_coef', 1e-6, 0.1, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 5.0),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    }
    # Build env
    envs = [make_env(i) for i in range(NUM_ENVS)]
    vec_env = DummyVecEnv(envs)
    # Instantiate model
    model = A2C(
        policy='MlpPolicy',
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=BASE_SEED,
        **params
    )
    # Short training for tuning
    model.learn(min(TOTAL_STEPS, 100_000))
    # Evaluate
    mean_reward, _ = evaluate_policy(model, vec_env,
                                     n_eval_episodes=EVAL_EPISODES,
                                     warn=False)
    vec_env.close()
    return mean_reward

# -----------------------------
# Tuning entry point
# -----------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tune A2C on HoverAviary')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--output', type=str,
                        default='tuning_results/a2c_hover_hp.json',
                        help='Output JSON for best params')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.trials)
    best = study.best_params.copy()
    with open(args.output, 'w') as f:
        json.dump(best, f, indent=2)
    print('Best hyperparameters:')
    for k, v in best.items(): print(f'  {k}: {v}')
