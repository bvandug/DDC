# ip_train_simulink.py
import os
from pathlib import Path
import numpy as np
import torch as th
import time  # Added for timing

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor  # <-- ADD THIS IMPORT

from ip_simulink_env import IPEnv


# ---------- config ----------
# Folder containing pendulum_core.slx, ip_reset.m, ip_step.m
THIS_DIR = Path(__file__).resolve().parent

MODEL_NAME = "pendulum_core"      # modified model
DT = 0.01
ANGLE_LIM = np.pi / 2
MAX_TIME = 5.0

# To-Workspace block *paths* (right-click block → Copy → Path)
ANGLE_BLOCK = "pendulum_core/To Workspace"
ANGVEL_BLOCK = "pendulum_core/To Workspace1"

TOTAL_STEPS = 100_000
LOGDIR = THIS_DIR / "trained_ip_models" / "A2C_IP_stepreset"
DEVICE = "cpu"



def make_env():
    # Create env; it starts its own MATLAB engine internally
    env = IPEnv(
        model=MODEL_NAME,
        dt=DT,
        angle_threshold=ANGLE_LIM,
        max_time=MAX_TIME,
        angle_block=ANGLE_BLOCK,
        angvel_block=ANGVEL_BLOCK,
        matlab_workdir=str(THIS_DIR),  # ensure MATLAB sees the .m files and model
    )
    env = Monitor(env) 
    return env


def main():
    os.makedirs(LOGDIR, exist_ok=True)

    # Single-process vec env (MATLAB Engine is not fork-safe)
    venv = DummyVecEnv([make_env])

    # --- Optimized Hyperparameters ---
    hyperparams = {
        "learning_rate": 0.000691421433969174,
        "gamma": 0.9199188292787973,
        "n_steps": 8,
        "ent_coef": 1.1634469351626833e-07,
        "vf_coef": 0.7832577554691627,
        "max_grad_norm": 1.269004043043889,
        "use_rms_prop": False,
        "layer_size": 263,
        "n_layers": 2,
    }
    
    policy_kwargs = {
        "activation_fn": th.nn.Tanh,
        "net_arch": [hyperparams["layer_size"]] * hyperparams["n_layers"],
    }
    # --------------------------------

    # Algo
    model = A2C(
        policy="MlpPolicy",
        env=venv,
        policy_kwargs=policy_kwargs,
        learning_rate=hyperparams["learning_rate"],
        gamma=hyperparams["gamma"],
        n_steps=hyperparams["n_steps"],
        ent_coef=hyperparams["ent_coef"],
        vf_coef=hyperparams["vf_coef"],
        max_grad_norm=hyperparams["max_grad_norm"],
        use_rms_prop=hyperparams["use_rms_prop"],
        verbose=1,
        tensorboard_log=str(LOGDIR),
        device=DEVICE,
    )

    # Train
    print("--- Starting Training ---")
    start_time = time.monotonic()
    model.learn(total_timesteps=TOTAL_STEPS)
    end_time = time.monotonic()
    print("--- Training Finished ---")

    # --- Report Training Time ---
    duration_s = end_time - start_time
    minutes = int(duration_s // 60)
    seconds = duration_s % 60
    print(f"Training took {minutes} minutes and {seconds:.2f} seconds.")
    # --------------------------

    model.save(LOGDIR / "A2C_IP_stepreset")

    # Clean shutdown
    venv.close()


if __name__ == "__main__":
    main()