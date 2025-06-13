import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from simulink_env import SimulinkEnv
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

if __name__ == "__main__":
    # Initialize the Simulink environment
    env = SimulinkEnv(model_name="PendCart", agent_block="PendCart/RL Agent", dt=0.01)
    timesteps = 500
    # , ("ppo", PPO), ("sac", SAC)
    for name, Algo in [("td3", TD3)]:
        print(f"Training {name} on Simulink model…")
        model = Algo(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=256,
        tau=0.005,
        train_freq=(1, "step"),
        policy_delay=2,
        action_noise=NormalActionNoise(mean=np.zeros(1), sigma=0.2 * np.ones(1)),
        target_policy_noise=0.2,
        target_noise_clip=0.5
        )

        # Add progress tracking
        print(f"Starting training for {timesteps} timesteps...")
        model.learn(total_timesteps=timesteps, progress_bar=True)
        print(f"Training completed. Total timesteps: {timesteps}")
        
        # Save the trained model
        model.save(f"{name}_simulinker500")
        print(f"Model saved to {name}_simulinker500")

    env.close()