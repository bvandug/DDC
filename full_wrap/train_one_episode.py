from stable_baselines3 import PPO, TD3
from simulink_env import SimulinkEnv  # or any dummy Gym env

# Create the model (randomly initialized weights)
model = PPO("MlpPolicy", SimulinkEnv(model_name="PendCart", dt=10))

model1 = TD3("MlpPolicy", SimulinkEnv(model_name="PendCart", dt=10))

# Immediately save it—no training at all!
model.save("ppo_simulinker")    # → writes ppo_vanilla.zip
model1.save("td3_simulink")