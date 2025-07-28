# ip_numpy_wrapper.py (NumPy-based gym wrapper)
import gym
from gym import spaces
import numpy as np
from typing import Optional
from ip_numpy import PendulumConfig, PendulumState, reset_pendulum_env, step_pendulum_env

class InvertedPendulumGymWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, config: Optional[PendulumConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.config = config if config else PendulumConfig()
        self.seed_val = seed if seed is not None else 0

        self.action_space = spaces.Box(
            low=-self.config.max_force,
            high=self.config.max_force,
            shape=(1,),
            dtype=np.float32,
        )

        high = np.array([np.pi, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state: Optional[PendulumState] = None

    def reset(self):
        self.state = reset_pendulum_env(seed=self.seed_val, config=self.config)
        return self._obs()

    def step(self, action):
        self.state, reward = step_pendulum_env(self.state, float(action), self.config)
        return self._obs(), float(reward), bool(self.state.done), {}

    def _obs(self):
        theta = self.state.theta
        theta_dot = self.state.theta_dot
        return np.array([theta, theta_dot], dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
