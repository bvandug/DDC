from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from simulink_env import SimulinkEnv

# Load environment and trained model
env = SimulinkEnv()
model = TD3.load("td3_simulinker")

# Evaluate the policy
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
