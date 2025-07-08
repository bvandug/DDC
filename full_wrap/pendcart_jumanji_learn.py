from jumanji.wrappers import JumanjiToGymWrapper
from full_wrap.ip_jax import PendCartEnv

env = PendCartEnv()
gym_env = JumanjiToGymWrapper(env, seed=42)

# Then pass gym_env into SB3
from stable_baselines3 import PPO

model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=100_000)
