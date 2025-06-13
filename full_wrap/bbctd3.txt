import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env

# Import the custom Simulink environment class from the other file
from BBCSimulink_env import BBCSimulinkEnv

# --- Main execution block ---
if __name__ == "__main__":
    # 1. Instantiate the Simulink Environment using the SB3 helper
    print("Starting the environment...")
    env = make_vec_env(BBCSimulinkEnv, n_envs=1)

    # 2. Define training parameters for the TD3 agent
    total_timesteps = 25000 
    
    # Define action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Create the TD3 model
    print(f"Creating the model...")
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(10, "step"), #Train every 10 steps
        action_noise=action_noise,
        verbose=1,
        tensorboard_log="./bbc_td3_tensorboard/"
    )

    # 3. Train the agent
    print(f"--- Starting Training {model.__class__.__name__} for {total_timesteps} Timesteps ---")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=1000 #Log to tensorboard every 1000 steps
    )
    print("--- Training Complete ---")

    # 4. Save the trained model
    model.save("td3_bbc_model")
    print("--- Model Saved as td3_bbc_model.zip ---")

    # 5. Close the environment (and the MATLAB engine)
    env.close()
