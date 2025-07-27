import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
from typing import Optional
from ip_jax import PendulumConfig, PendulumState, reset_pendulum_env, step_pendulum_env

class InvertedPendulumGymWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        config: Optional[PendulumConfig] = None,
        seed: Optional[int] = None,
        noise: bool = False,
        noise_std: float = 0.01,
    ):
        super().__init__()
        self.config    = config if config else PendulumConfig()
        self.rng       = jax.random.PRNGKey(seed if seed is not None else 0)
        self.noise     = noise
        self.noise_std = noise_std
        self.np_rng    = np.random.RandomState(int(seed or 0))

        # Action space now represents torque, not force
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
        if seed is not None:
            self.rng    = jax.random.PRNGKey(seed)
            self.np_rng = np.random.RandomState(int(seed))
        self.rng, subkey = jax.random.split(self.rng)
        self.state = reset_pendulum_env(subkey, self.config)
        obs = self._obs()
        if self.noise:
            obs = obs + self.np_rng.normal(0, self.noise_std, size=obs.shape)
        return obs, {}

    def step(self, action):
        # Interpret action as torque
        self.state, reward = step_pendulum_env(self.state, float(action), self.config)
        terminated = bool(self.state.done)
        truncated = False
        obs = self._obs()
        if self.noise:
            obs = obs + self.np_rng.normal(0, self.noise_std, size=obs.shape)
        return obs, float(reward), terminated, truncated, {}

    def _obs(self):
        theta     = self.state.theta
        theta_dot = self.state.theta_dot
        return np.array([theta, theta_dot], dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass

class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, torque_values):
        super().__init__(env)
        # Discrete set of allowable torques
        self.torque_values = np.asarray(torque_values, dtype=np.float32)
        self.action_space  = spaces.Discrete(len(self.torque_values))

    def action(self, act_idx):
        # Map discrete index to continuous torque
        return np.array([self.torque_values[int(act_idx)]], dtype=np.float32)
