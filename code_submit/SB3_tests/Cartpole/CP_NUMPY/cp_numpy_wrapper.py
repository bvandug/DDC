import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional

from cp_numpy import CartPoleConfig, CartPoleState, reset_cartpole_env, step_cartpole_env

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
        """ Initialize a Gymnasium wrapper around the NumPy cart-pole.

            Sets up RNGs, action/observation spaces, optional Gaussian observation
            noise, partial vs. full observations, and a `max_steps` cap that matches
            `max_episode_time / dt` to avoid floating-point drift.

            Parameters
            ----------
            config : CartPoleConfig | None, optional
                Environment configuration. Defaults to `CartPoleConfig()`.
            seed : int | None, optional
                Seed for the internal RNGs.
            noise : bool, optional
                If True, add zero-mean Gaussian noise to observations.
            noise_std : float, optional
                Standard deviation for the observation noise.
            partial_obs : bool, optional
                If True, observations are `[theta, theta_dot]` (shape (2,));
                otherwise `[x, x_dot, theta, theta_dot]` (shape (4,)).
        """

        super().__init__()
        self.config     = config if config else CartPoleConfig()
        self.rng        = np.random.default_rng(seed)
        self.noise      = noise
        self.noise_std  = noise_std
        self.partial_obs = partial_obs
        self.np_rng     = np.random.default_rng(seed)
        # limit episodes to exact step count to avoid float drift
        self.max_steps  = int(self.config.max_episode_time / self.config.dt)

        # Continuous force action space
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
        """ Reset the environment and return the initial observation.

            Resets the step counter, reseeds RNGs if a per-episode `seed` is given,
            derives an integer seed for the angle initialization, and returns the
            (obtionally noisy) observation.

            Parameters
            ----------
            seed : int | None, optional
                Per-episode seed (overrides constructor seed for this reset).
            options : dict | None, optional
                Unused Gymnasium options placeholder.

            Returns
            -------
            tuple[np.ndarray, dict]
                `(obs, info)` where `obs` is either:
                - partial: `[theta, theta_dot]` (rad, rad/s), or
                - full: `[x, x_dot, theta, theta_dot]` (m, m/s, rad, rad/s).
                `info` is an empty dict.
        """

        self.step_count = 0
        if seed is not None:
            self.rng    = np.random.default_rng(seed)
            self.np_rng = np.random.default_rng(seed)
        theta_seed = self.rng.integers(0, 2**32 - 1)
        self.state = reset_cartpole_env(theta_seed, self.config)
        obs = self._obs()
        if self.noise:
            obs = obs + self.np_rng.normal(0, self.noise_std, size=obs.shape)
        return obs, {}

    def step(self, action):
        """Advance one step and return the Gymnasium 5-tuple.

            Applies the control, steps the dynamics, increments a strict step
            counter, and returns observation (optionally noisy), reward, terminal/
            truncation flags, and an empty info dict.

            Parameters
            ----------
            action :
                Control input (N). Scalars or (1,) arrays are accepted.

            Returns
            -------
            tuple
                `(obs, reward, terminated, truncated, info)` where:
                - `obs` : np.ndarray (same layout as in `reset`)
                - `reward` : float (typically `cos(theta)` from the new state)
                - `terminated` : bool (pole fall / time limit from dynamics)
                - `truncated` : bool (True if `step_count >= max_steps`)
                - `info` : dict (empty)
        """

        self.step_count += 1
        self.state, reward = step_cartpole_env(self.state, float(action), self.config)
        # pole-fall termination
        terminated = bool(self.state.done)
        truncated = (self.step_count >= self.max_steps)
        obs = self._obs()
        if self.noise:
            obs = obs + self.np_rng.normal(0, self.noise_std, size=obs.shape)
        return obs, float(reward), terminated, truncated, {}

    def _obs(self):
        """ Build the current observation vector from the internal state.

            Returns
            -------
            np.ndarray
                If `partial_obs` is True: `[theta, theta_dot]` (rad, rad/s).
                Otherwise: `[x, x_dot, theta, theta_dot]` (m, m/s, rad, rad/s).
        """

        if self.partial_obs:
            return np.array([self.state.theta, self.state.theta_dot], dtype=np.float32)
        else:
            return np.array([
                self.state.x,
                self.state.x_dot,
                self.state.theta,
                self.state.theta_dot
            ], dtype=np.float32)

    def get_internal_state(self):
        """ Return the underlying `CartPoleState` without copying.

            Returns
            -------
            CartPoleState
                The current internal state (useful for diagnostics).
        """

        return self.state

    def render(self, mode="human"):
        """ Print a one-line, human-readable snapshot of the current state.

            Parameters
            ----------
            mode : str, optional
                Unused; present for Gymnasium compatibility.
        """

        print(
            f"[t = {self.state.t:.2f}s] x = {self.state.x:.2f} m | x_dot = {self.state.x_dot:.2f} m/s | "
            f"theta = {np.rad2deg(self.state.theta):.1f}° | theta_dot = {np.rad2deg(self.state.theta_dot):.1f}°/s"
        )

    def close(self):
        """Close the environment (no external resources to release)."""

        pass

class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, force_values):
        """ Wrap a continuous-action env with a discrete index mapping.

            Parameters
            ----------
            env :
                Base environment with a continuous action space of shape (1,).
            force_values :
                Sequence of scalar controls; index `i` maps to `force_values[i]`.

            Notes
            -----
            Sets `action_space = Discrete(len(force_values))`.
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