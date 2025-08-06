import os
import torch.nn as nn
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# === Configuration (hard-coded parameters) ===
NUM_ENVS = 4
TOTAL_TIMESTEPS = 500_000
DRONE_MODEL = DroneModel.CF2X
PHYSICS = Physics.PYB
USE_GUI = True
SEED = 42

# A2C Hyperparameters tuned for a larger network
N_STEPS       = 256       # more steps per update for stable gradients
GAMMA         = 0.995     # high discount for long-horizon hover
ENT_COEF      = 0.001     # small entropy bonus to aid exploration
VF_COEF       = 0.5       # balance policy vs value loss
MAX_GRAD_NORM = 0.5       # tighter gradient clipping for stability
LEARNING_RATE = 7e-4      # moderate learning rate

# Larger network architecture
# 4 hidden layers: 512 → 512 → 256 → 128, with ReLU activations
policy_kwargs = dict(
    net_arch=dict(pi=[512, 512, 256, 128], vf=[512, 512, 256, 128]),
    activation_fn=nn.ReLU
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
