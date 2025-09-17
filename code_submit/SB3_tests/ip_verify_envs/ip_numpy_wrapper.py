# ip_numpy_wrapper.py (NumPy-based gym wrapper)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from ip_numpy import PendulumConfig, PendulumState, reset_pendulum_env, step_pendulum_env

class InvertedPendulumGymWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, config: Optional[PendulumConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.config = config if config else PendulumConfig()
        
        # Initialize the random number generator for the environment
        # This will be used to generate seeds for the reset function
        self.np_rng = np.random.RandomState(seed if seed is not None else 0)

        self.action_space = spaces.Box(
            low=-self.config.max_torque,
            high=self.config.max_torque,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: [theta, theta_dot]
        high = np.array([np.pi, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state: Optional[PendulumState] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to a new random initial state.

        This method now follows the standard Gymnasium API. It re-seeds the
        environment's random number generator if a new seed is provided,
        ensuring reproducibility while allowing for random initializations
        during training.
        """
        super().reset(seed=seed)
        if seed is not None:
            #self.np_rng = np.random.RandomState(seed)
            self.np_rng = np.random.default_rng(seed)
        #new_seed = self.np_rng.randint(0, 2**32 - 1)
        new_seed = self.np_rng.integers(0, 2**32 - 1, dtype=np.uint32)

        self.state = reset_pendulum_env(seed=new_seed, config=self.config)
        
        return self._obs(), {}

    def step(self, action):
        """
        Takes a step in the environment.
        
        The return signature is updated to match the Gymnasium standard.
        """
        
        self.state, reward = step_pendulum_env(self.state, float(action), self.config)
        terminated = bool(self.state.done)
        truncated = False
        obs = self._obs()
        
        # Return 5 values as per the Gymnasium API
        return obs, float(reward), terminated, truncated, {}

    def _obs(self):
        """Helper function to get the observation from the state."""
        theta = self.state.theta
        theta_dot = self.state.theta_dot
        return np.array([theta, theta_dot], dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
