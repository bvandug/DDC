import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Import the custom Simulink environment class from the other file
# Make sure 'BBCSimulink_env.py' is in the same directory
from BBCSimulink_env import BBCSimulinkEnv

# --- Definition of the Custom Callback ---
class TimestepAndFinalStatsCallback(BaseCallback):
    """
    Custom callback that logs SB3 stats at a specific timestep frequency
    and also reports final stats at the end of training.
    """
    def __init__(self, log_freq: int, verbose: int = 0):
        super(TimestepAndFinalStatsCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        """
        # Log the standard SB3 output at the specified frequency
        if self.n_calls % self.log_freq == 0:
            # Dumps all the values collected by the logger to the console
            self.model.logger.dump(step=self.num_timesteps)

        # Accumulate reward for the final, incomplete episode report
        # self.locals['rewards'] is a numpy array with the reward of the last step
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward

        # Reset the reward tracker if an episode has ended
        done = self.locals['dones'][0]
        if done:
            self.current_episode_reward = 0.0

        return True # Return True to continue training

    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        """
        print("\n--- Training Complete ---")

        # Dump the very last set of logs
        self.model.logger.dump(step=self.num_timesteps)

        # Report the reward from the final, potentially incomplete, episode
        print(f"\nReward accumulated in the final (incomplete) episode: {self.current_episode_reward:.4f}")
        print("-----------------------")


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
    print("Creating the model...")
    # Set verbose=0 to prevent SB3 from logging on its own (our callback handles it)
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
        verbose=0, 
    )

    # Instantiate the custom callback to log every 1000 timesteps
    custom_callback = TimestepAndFinalStatsCallback(log_freq=1000)

    # 3. Train the agent
    print(f"--- Starting Training {model.__class__.__name__} for {total_timesteps} Timesteps ---")
    # The callback is passed to the learn method
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=custom_callback
    )
    # The callback now handles the "Training Complete" message and final stats

    # 4. Save the trained model
    model.save("td3_bbc_model")
    print("--- Model Saved as td3_bbc_model.zip ---")

    # 5. Close the environment (and the MATLAB engine)
    env.close()
