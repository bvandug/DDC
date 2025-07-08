# train_sb3_pendulum.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ip_jax_wrapper import InvertedPendulumGymWrapper

# Create wrapped environment
def make_env():
    return InvertedPendulumGymWrapper()

env = DummyVecEnv([make_env])

# Initialize and train PPO
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./pendulum_logs/")
model.learn(total_timesteps=100_000)

# Save model
model.save("ppo_inverted_pendulum")
print("✅ Model trained and saved.")
