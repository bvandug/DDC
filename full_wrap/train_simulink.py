import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from simulink_env import SimulinkEnv
from stable_baselines3 import PPO, SAC, TD3

if __name__ == "__main__":
    # Initialize the Simulink environment
    env = SimulinkEnv(model_name="PendCart", agent_block="PendCart/RL Agent", dt=0.1)
    timesteps = 300

    for name, Algo in [("td3", TD3), ("ppo", PPO), ("sac", SAC)]:
        print(f"Training {name} on Simulink model…")
        model = Algo("MlpPolicy", env, verbose=1)
        # Plain learn, no progress bar
        model.learn(total_timesteps=timesteps)
        # Save the trained model
        model.save(f"{name}_simulink")

    env.close()
