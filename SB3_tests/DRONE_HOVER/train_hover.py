import os
import torch.nn as nn
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# === Configuration (hard-coded parameters) ===
NUM_ENVS = 8
TOTAL_TIMESTEPS = 100_000
DRONE_MODEL = DroneModel.CF2X
PHYSICS = Physics.PYB
USE_GUI = True
SEED = 42

# A2C Hyperparameters tuned for a 3-layer network (logical defaults):
# - Larger batch for stability: n_steps = 256
# - Emphasize long-horizon control: gamma = 0.995
# - Light exploration bonus: ent_coef = 0.001
# - Standard value loss weight: vf_coef = 0.5
# - Tighter gradient clipping: max_grad_norm = 0.5
# - Moderate learning rate: learning_rate = 3e-4
N_STEPS       = 256
GAMMA         = 0.995
ENT_COEF      = 0.001
VF_COEF       = 0.5
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 3e-4
# Network architecture from best trial: single hidden layer of width 504, Tanh activation
policy_kwargs = dict(
    net_arch=dict(pi=[256,256], vf=[256,256]),
    activation_fn=nn.Tanh
)

# Logging & save dirs
TENSORBOARD_LOG = "./tensorboard"
SAVE_DIR = "./models"

# === Environment factory ===
def make_env(rank: int):
    """
    Creates a Monitor-wrapped HoverAviary environment with a fixed seed.
    Only the first environment opens a GUI.
    """
    def _init():
        env = HoverAviary(
            drone_model=DRONE_MODEL,
            physics=PHYSICS,
            gui=(USE_GUI and rank == 0)
        )
        env = Monitor(env)
        env.reset(seed=SEED + rank)
        return env
    return _init

if __name__ == "__main__":
    # Create vectorized environments
    env_fns = [make_env(i) for i in range(NUM_ENVS)]
    vec_env = DummyVecEnv(env_fns)

    # Instantiate A2C model with tuned parameters
    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=N_STEPS,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        learning_rate=LEARNING_RATE,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG,
        seed=SEED
    )

    # Train the agent with more frequent logging
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=10,  # prints every 10 updates
    )

    # Save the trained model
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "a2c_drone_hover")
    model.save(save_path)
    print(f"Model saved to: {save_path}")
