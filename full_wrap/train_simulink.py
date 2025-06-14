import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from simulink_env import SimulinkEnv
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

if __name__ == "__main__":
    # Initialize the Simulink environment
    env = SimulinkEnv(model_name="PendCart", agent_block="PendCart/RL Agent", dt=0.01)
    timesteps = 50000

    for name, Algo in [("td3", TD3)]:
        print(f"Training {name} on Simulink model…")
        
        # Construct the action noise with the optimized sigma
        action_noise = NormalActionNoise(
            mean=np.zeros(1),
            sigma=0.1020777897320515 * np.ones(1)
        )

        model = Algo(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0009531916300780667,
            buffer_size=127040,
            batch_size=511,
            tau=0.007702202735525895,
            gamma=0.9428889303152354,
            train_freq=(10, "step"),
            policy_delay=4,
            action_noise=action_noise,
            target_policy_noise=0.10477469465730704,
            target_noise_clip=0.5635043295165336
        )

        # Train
        model.learn(total_timesteps=timesteps)
        # Save the trained model
        model.save(f"{name}_t11")

    env.close()
