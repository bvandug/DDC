from stable_baselines3 import PPO, TD3, SAC
from simulink_env import SimulinkEnv  # or any dummy Gym env

# # Create the model (randomly initialized weights)
model = PPO("MlpPolicy", SimulinkEnv(model_name="PendCart", dt=0.01))
model.save("ppo_simulinker")  

model1 = TD3("MlpPolicy", SimulinkEnv(model_name="PendCart", dt=0.01))
model1.save("td3_simulinker")

model2 = SAC("MlpPolicy", SimulinkEnv(model_name="PendCart", dt=0.01))
model2.save("sac_simulinker") 