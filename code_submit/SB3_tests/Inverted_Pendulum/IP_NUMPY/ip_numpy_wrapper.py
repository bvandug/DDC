"""Gymnasium wrapper for NumPy-based inverted pendulum environment.

Provides a `gym.Env`-compatible interface around the NumPy physics
functions, allowing use with Stable-Baselines3 or other RL libraries.
Implements reset/step methods per Gymnasium API.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
from ip_numpy import PendulumConfig, PendulumState, reset_pendulum_env, step_pendulum_env

class InvertedPendulumGymWrapper(gym.Env):
    """
    Gymnasium-compatible environment for NumPy pendulum dynamics.

    Args:
        config (PendulumConfig, optional): Custom pendulum parameters.
        seed (int, optional): Random seed for reproducibility.

    Attributes:
        action_space (spaces.Box): Continuous torque space [-max, +max].
        observation_space (spaces.Box): Observations = [theta, theta_dot].
        state (PendulumState): Current state of the pendulum.
        np_rng (np.random.Generator): RNG for seeding and reset reproducibility.
    """
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
        Reset environment state and sample a new initial angle.

        If a seed is provided, reseeds the internal RNG before generating
        a new random initial state for the pendulum.

        Args:
            seed (int, optional): Seed to reinitialize RNG.
            options (dict, optional): Reserved for API compliance.

        Returns:
            tuple[np.ndarray, dict]: Observation array and empty info dict.
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
        Advance simulation by one step using the given action.

        Args:
            action (float): Torque to apply to the pendulum.

        Returns:
            tuple:
                - obs (np.ndarray): [theta, theta_dot] observation.
                - reward (float): Cosine-based reward from next state.
                - terminated (bool): True if episode terminated (angle/time).
                - truncated (bool): Always False (no time truncation used).
                - info (dict): Empty dictionary for API compliance.
        """
        self.state, reward = step_pendulum_env(self.state, float(action), self.config)
        terminated = bool(self.state.done)
        truncated = False
        obs = self._obs()
        
        # Return 5 values as per the Gymnasium API
        return obs, float(reward), terminated, truncated, {}

    def _obs(self):
        """Return the current observation vector [theta, theta_dot].

        Returns:
            np.ndarray: Observation as float32 array.
        """
        theta = self.state.theta
        theta_dot = self.state.theta_dot
        return np.array([theta, theta_dot], dtype=np.float32)

    def render(self, mode="human"):
        """Render the environment (currently not implemented)."""
        pass

    def close(self):
        """Close environment resources (currently no-op)."""
        pass
