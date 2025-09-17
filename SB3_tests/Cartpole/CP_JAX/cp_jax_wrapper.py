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
        """Initialize a Gymnasium wrapper around the JAX cart-pole.

            Configures action/observation spaces, seeds RNGs, and enables optional
            Gaussian observation noise and partial observations.

            Parameters
            ----------
            config : CartPoleConfig | None, optional
                Environment configuration. Uses default `CartPoleConfig()` if None.
            seed : int | None, optional
                Seed for both JAX and NumPy RNGs.
            noise : bool, optional
                If True, add Gaussian noise to observations.
            noise_std : float, optional
                Standard deviation of the observation noise.
            partial_obs : bool, optional
                If True, observations are `[theta, theta_dot]` (shape (2,));
                otherwise full state `[x, x_dot, theta, theta_dot]` (shape (4,)).
        """

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
        """Reset the environment and return the initial observation.

            Reseeds RNGs if a `seed` is provided, samples a new initial state via
            `reset_cartpole_env`, and optionally adds Gaussian noise to the
            observation.

            Parameters
            ----------
            seed : int | None, optional
                Per-episode seed (overrides the constructor seed for this reset).
            options : dict | None, optional
                Unused Gymnasium options placeholder.

            Returns
            -------
            tuple[np.ndarray, dict]
                `(obs, info)` where `obs` is either:
                - partial: `[theta, theta_dot]` (radians, rad/s), or
                - full: `[x, x_dot, theta, theta_dot]` (m, m/s, rad, rad/s).
                `info` is an empty dict.
        """

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
        """Advance one step using `step_cartpole_env` and return Gymnasium tuple.

            Applies the control, updates the internal JAX state, computes reward,
            and returns observation (optionally noisy), reward, termination flags,
            and an empty info dict.

            Parameters
            ----------
            action :
                Control input (N). Scalars or (1,) arrays are accepted.

            Returns
            -------
            tuple
                `(obs, reward, terminated, truncated, info)` where:
                - `obs` : np.ndarray, same layout as in `reset`.
                - `reward` : float, typically `cos(theta)` from the new state.
                - `terminated` : bool, True if the angle threshold/time limit hit.
                - `truncated` : bool, always False here (no external truncation).
                - `info` : dict, empty.
        """

        self.state, reward = step_cartpole_env(self.state, float(action), self.config)
        terminated = bool(self.state.done)
        truncated = False
        obs = self._obs()
        if self.noise:
            obs += self.np_rng.normal(0, self.noise_std, size=obs.shape)
        return obs, float(reward), terminated, truncated, {}

    def _obs(self):
        """Build the current observation vector from the internal state.

            Returns
            -------
            np.ndarray
                If `partial_obs` is True: `[theta, theta_dot]` (rad, rad/s).
                Otherwise: `[x, x_dot, theta, theta_dot]` (m, m/s, rad, rad/s).
        """

        if self.partial_obs:
            return np.array([self.state.theta, self.state.theta_dot], dtype=np.float32)
        else:
            return np.array([self.state.x, self.state.x_dot, self.state.theta, self.state.theta_dot], dtype=np.float32)

    def get_internal_state(self):
        """Return the underlying JAX `CartPoleState` without copying.

            Returns
            -------
            CartPoleState
                The current internal state (may be used for diagnostics).
        """

        return self.state

    def render(self, mode="human"):
        """Print a human-readable single-line snapshot of the current state.

            Parameters
            ----------
            mode : str, optional
                Unused; present for Gymnasium compatibility.

            Notes
            -----
            Prints to stdout with units (m, m/s, deg, deg/s).
        """

        print(
            f"[t = {self.state.t:.2f}s] x = {self.state.x:.2f} m | ẋ = {self.state.x_dot:.2f} m/s | "
            f"θ = {np.rad2deg(self.state.theta):.1f}° | θ̇ = {np.rad2deg(self.state.theta_dot):.1f}°/s"
        )

    def close(self):
        pass


class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, force_values):
        """Wrap a continuous-action env with a discrete action index mapping.

            Parameters
            ----------
            env :
                The base environment with a continuous action space of shape (1,).
            force_values :
                Sequence of scalar controls; index `i` maps to `force_values[i]`.

            Notes
            -----
            Sets `action_space` to `Discrete(len(force_values))`.
        """

        super().__init__(env)
        self.force_values = np.asarray(force_values, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.force_values))

    def action(self, act_idx):
        """Map a discrete index to a (1,) float32 continuous action.

            Parameters
            ----------
            act_idx :
                Discrete action index selected by the policy.

            Returns
            -------
            np.ndarray
                A one-element float32 array containing the mapped force value.
        """

        return np.array([self.force_values[int(act_idx)]], dtype=np.float32)
