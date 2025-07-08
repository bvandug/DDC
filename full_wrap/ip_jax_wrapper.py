# jax_pendulum_gym_wrapper.py

import gym
from gym import spaces
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional
from ip_jax import PendulumConfig, PendulumState, reset_pendulum_env, step_pendulum_env

class InvertedPendulumGymWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, config: Optional[PendulumConfig] = None, seed: Optional[int] = None):
        super().__init__()
        self.config = config if config else PendulumConfig()
        self.rng = jax.random.PRNGKey(seed if seed is not None else 0)

        # Continuous action and observation space
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
        self.rng, subkey = jax.random.split(self.rng)
        self.state = reset_pendulum_env(subkey, self.config)
        return np.array([self.state.theta, self.state.theta_dot], dtype=np.float32)

    def step(self, action):
        self.state, reward = step_pendulum_env(self.state, float(action), self.config)
        obs = np.array([self.state.theta, self.state.theta_dot], dtype=np.float32)
        done = bool(self.state.done)
        return obs, float(reward), done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass
