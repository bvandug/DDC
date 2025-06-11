import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from simulink_env import SimulinkEnv
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from tqdm import tqdm

def make_env():
    def _init():
        env = SimulinkEnv(model_name="PendCart", agent_block="PendCart/RL Agent", dt=0.01)
        return env
    return _init

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting A2C Training Setup")
    print("="*50)
    
    # Number of parallel environments
    n_envs = 1
    print(f"\nCreating {n_envs} parallel environments...")
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    print("Vectorized environment created successfully.")
    
    # Training parameters - reduced timesteps
    timesteps = 10000  # Reduced from 50000 to 10000
    print(f"\nTraining will run for {timesteps} timesteps")
    
    # A2C specific parameters optimized for faster learning
    print("\nConfiguring A2C parameters...")
    a2c_params = {
        "learning_rate": 2e-3,        # Increased for faster learning
        "n_steps": 5,                 # Keep short rollout length
        "gamma": 0.99,                # Standard discount factor
        "gae_lambda": 0.95,           # Good balance for advantage estimation
        "ent_coef": 0.01,             # Encourages exploration
        "vf_coef": 0.5,              # Balances policy and value function
        "max_grad_norm": 0.5,         # Prevents too large policy updates
        "rms_prop_eps": 1e-5,         # Standard RMSprop epsilon
        "use_rms_prop": True,         # Use RMSprop optimizer
        "use_sde": False,             # No state-dependent exploration
        "sde_sample_freq": -1,        # Not using SDE
        "normalize_advantage": False,  # Don't normalize advantages
        "tensorboard_log": "./a2c_tensorboard/"  # For monitoring training
    }
    
    print("\nInitializing A2C model...")
    model = A2C(
        "MlpPolicy", 
        env,
        verbose=1,
        **a2c_params
    )
    print("Model initialized successfully.")

    # Add progress tracking
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    print(f"\nTraining for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    print("\nTraining completed successfully!")
    
    # Save the trained model
    print("\nSaving trained model...")
    model.save("a2c_simulinker")
    print("Model saved as 'a2c_simulinker'")

    print("\nCleaning up...")
    env.close()
    print("Training process completed successfully!\n")
