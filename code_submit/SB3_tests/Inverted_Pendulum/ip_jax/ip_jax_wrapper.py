"""Gymnasium wrapper for the JAX-based pendulum environment.

Provides a `gym.Env`-compatible interface with continuous or discretized
torque control, optional observation noise, and reproducible seeding.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import jax
from typing import Optional
from ip_jax import PendulumConfig, PendulumState, reset_pendulum_env, step_pendulum_env

class InvertedPendulumGymWrapper(gym.Env):
    """Gymnasium-compatible environment for the inverted pendulum.

    This wrapper exposes the JAX pendulum simulator as a `gym.Env`,
    making it easy to use with RL libraries (e.g., SB3, RLlib).

    Args:
        config (PendulumConfig, optional): Custom environment configuration.
        seed (int, optional): Random seed for reproducibility.
        noise (bool): Whether to add Gaussian noise to observations.
        noise_std (float): Standard deviation of observation noise.

    Attributes:
        action_space (spaces.Box): Continuous torque space [-max_torque, +max_torque].
        observation_space (spaces.Box): State vector [theta, theta_dot].
        state (PendulumState): Current simulator state.
        rng: JAX PRNGKey for reproducible resets.
        np_rng: NumPy RNG for noise sampling.
    """
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
        """Reset the environment state at the start of an episode.

        Splits the JAX PRNGKey, samples a new initial angle, and
        optionally injects Gaussian noise into the observation.

        Args:
            seed (int, optional): Seed for reproducibility of resets.
            options (dict, optional): Unused, kept for API compliance.

        Returns:
            tuple[np.ndarray, dict]: Observation array and empty info dict.
        """
        if seed is not None: #seed code
            self.rng    = jax.random.PRNGKey(seed)
            self.np_rng = np.random.RandomState(int(seed))
        self.rng, subkey = jax.random.split(self.rng)
        self.state = reset_pendulum_env(subkey, self.config)
        obs = self._obs()
        if self.noise:
            obs = obs + self.np_rng.normal(0, self.noise_std, size=obs.shape)
        return obs, {}

    def step(self, action):
        """Advance the environment by one time step.

        Applies the given torque, steps the physics, and returns the new state,
        reward, and termination flags.

        Args:
            action (array-like): Torque to apply (shape (1,), float).

        Returns:
            tuple:
                - obs (np.ndarray): Next observation [theta, theta_dot].
                - reward (float): Reward after this step.
                - terminated (bool): True if episode ended by angle/time.
                - truncated (bool): Always False (no time limit here).
                - info (dict): Empty dict for API compliance.
        """
        self.state, reward = step_pendulum_env(self.state, float(action), self.config)
        terminated = bool(self.state.done)
        truncated = False
        obs = self._obs()
        if self.noise:
            obs = obs + self.np_rng.normal(0, self.noise_std, size=obs.shape) #add guassian noise
        return obs, float(reward), terminated, truncated, {}

    def _obs(self):
        """Construct observation vector from current state.

        Returns:
            np.ndarray: [theta, theta_dot] as float32 array.
        """
        theta     = self.state.theta
        theta_dot = self.state.theta_dot
        return np.array([theta, theta_dot], dtype=np.float32)

    def render(self, mode="human"):
        """Render the environment (not implemented).

        Currently a no-op placeholder to satisfy Gymnasium API.
        """
        pass

    def close(self):
        """Clean up resources (not implemented)."""
        pass

class DiscretizedActionWrapper(gym.ActionWrapper):
    """Action wrapper that maps discrete indices to torque values.

    Converts a continuous-control environment into a discrete one by
    restricting the action space to a fixed set of torque values.

    Args:
        env (gym.Env): Environment to wrap.
        torque_values (array-like): Discrete torque values allowed.

    Attributes:
        torque_values (np.ndarray): Array of allowed torques.
        action_space (spaces.Discrete): Discrete action space [0..n-1].
    """
    def __init__(self, env, torque_values):
        super().__init__(env)
        # Discrete set of allowable torques
        self.torque_values = np.asarray(torque_values, dtype=np.float32)
        self.action_space  = spaces.Discrete(len(self.torque_values))

    def action(self, act_idx):
        """Convert a discrete action index to a continuous torque value.
        Args:
            act_idx (int): Index into `torque_values`.

        Returns:
            np.ndarray: Single-element array containing the chosen torque.
        """
        # Map discrete index to continuous torque
        return np.array([self.torque_values[int(act_idx)]], dtype=np.float32)
