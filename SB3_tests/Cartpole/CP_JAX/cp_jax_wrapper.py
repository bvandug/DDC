import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional

# Directly import CartPole logic
from cp_jax import (
    CartPoleConfig,
    CartPoleState,
    reset_cartpole_env,
    step_cartpole_env,
)

class CartPoleGymWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        config: Optional[CartPoleConfig] = None,
        seed: Optional[int] = None,
        noise: bool = False,
        noise_std: float = 0.01,
        partial_obs: bool = True,
    ):
        super().__init__()
        self.config    = config if config else CartPoleConfig()
        self.rng       = jax.random.PRNGKey(seed if seed is not None else 0)
        self.noise     = noise
        self.noise_std = noise_std
        self.partial_obs = partial_obs
        self.np_rng    = np.random.RandomState(int(seed or 0))

        self.action_space = spaces.Box(
            low=-self.config.max_force,
            high=self.config.max_force,
            shape=(1,),
            dtype=np.float32,
        )

        obs_dim = 2 if self.partial_obs else 4
        high = np.array([np.pi] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.state: Optional[CartPoleState] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng    = jax.random.PRNGKey(seed)
            self.np_rng = np.random.RandomState(int(seed))
        self.rng, subkey = jax.random.split(self.rng)
        self.state = reset_cartpole_env(subkey, self.config)
        obs = self._obs()
        if self.noise:
            obs += self.np_rng.normal(0, self.noise_std, size=obs.shape)
        return obs, {}

    def step(self, action):
        self.state, reward = step_cartpole_env(self.state, float(action), self.config)
        terminated = bool(self.state.done)
        truncated = False
        obs = self._obs()
        if self.noise:
            obs += self.np_rng.normal(0, self.noise_std, size=obs.shape)
        return obs, float(reward), terminated, truncated, {}

    def _obs(self):
        if self.partial_obs:
            return np.array([self.state.theta, self.state.theta_dot], dtype=np.float32)
        else:
            # return np.array([self.state.theta, self.state.theta_dot], dtype=np.float32)
            return np.array([self.state.x, self.state.x_dot, self.state.theta, self.state.theta_dot], dtype=np.float32)

    def get_internal_state(self):
        return self.state

    def render(self, mode="human"):
        print(
            f"[t = {self.state.t:.2f}s] x = {self.state.x:.2f} m | ẋ = {self.state.x_dot:.2f} m/s | "
            f"θ = {np.rad2deg(self.state.theta):.1f}° | θ̇ = {np.rad2deg(self.state.theta_dot):.1f}°/s"
        )

    def close(self):
        pass


class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, force_values):
        super().__init__(env)
        self.force_values = np.asarray(force_values, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.force_values))

    def action(self, act_idx):
        return np.array([self.force_values[int(act_idx)]], dtype=np.float32)
