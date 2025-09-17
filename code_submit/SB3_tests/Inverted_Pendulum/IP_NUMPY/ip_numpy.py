"""NumPy-based pendulum dynamics for SB3 training.

Provides a lightweight, deterministic alternative to the JAX
version of the pendulum environment. Includes state container,
physics integration, reset logic using JAX RNG for reproducibility,
and a simple cosine-based reward.
"""

import numpy as np
from typing import NamedTuple
import jax
import jax.numpy as jnp

class PendulumConfig(NamedTuple):
    """Configuration parameters for the NumPy pendulum.

    Attributes:
        m (float): Pendulum mass in kg.
        L (float): Length from pivot to center of mass in meters.
        g (float): Gravitational acceleration (m/s²).
        dt (float): Simulation time step in seconds.
        angle_threshold (float): Max allowed |theta| before termination (rad).
        max_torque (float): Maximum torque magnitude (N·m).
        max_episode_time (float): Episode time limit (seconds).
    """
    m: float = 0.2
    L: float = 0.15
    g: float = 9.8
    dt: float = 0.01
    angle_threshold: float = np.pi / 2
    max_torque: float = 2.0
    max_episode_time: float = 5.0

class PendulumState(NamedTuple):
    """Container for pendulum state variables.

    Attributes:
        theta (float): Angular position in radians.
        theta_dot (float): Angular velocity (rad/s).
        t (float): Elapsed simulation time (seconds).
        done (bool): Termination flag for episode.
    """
    theta: float
    theta_dot: float
    t: float
    done: bool

def angle_normalize(x):
    """Normalize an angle to the range [-π, π].

    Args:
        x (float): Input angle in radians.

    Returns:
        float: Normalized angle wrapped into [-π, π].
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def pendulum_dynamics(state: PendulumState, action: float, config: PendulumConfig) -> PendulumState:
    """Compute next state using Euler integration of pendulum dynamics.

    Args:
        state (PendulumState): Current state (theta, theta_dot, t).
        action (float): Applied torque (clipped to ±max_torque).
        config (PendulumConfig): Physical constants and integration settings.

    Returns:
        PendulumState: Updated state with normalized theta and done flag.
    """
    tau = np.clip(action, -config.max_torque, config.max_torque)

    theta = state.theta
    theta_dot = state.theta_dot
    m, L, g = config.m, config.L, config.g

    I = m * L ** 2

    theta_ddot = (-m * g * L * np.sin(theta) + tau) / I

    theta_dot_new = theta_dot + theta_ddot * config.dt
    theta_new = theta + theta_dot_new * config.dt
    t_new = state.t + config.dt

    done = abs(theta_new) > config.angle_threshold or state.t + 1e-9 >= config.max_episode_time

    return PendulumState(
        theta=angle_normalize(theta_new),
        theta_dot=theta_dot_new,
        t=t_new,
        done=done
    )

def reset_pendulum_env(seed, config: PendulumConfig) -> PendulumState:
    """Initialize pendulum state with random theta.

    Samples theta uniformly from [-1, 1] rad, with small-offset adjustment
    if close to zero, ensuring meaningful starting conditions.

    Args:
        seed (int): Random seed for reproducibility.
        config (PendulumConfig): Environment configuration.

    Returns:
        PendulumState: Reset state with zero velocity and time.
    """
    key = jax.random.PRNGKey(seed)
    theta = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    theta = jnp.where(jnp.abs(theta) < 0.05, theta + 0.1, theta)
    return PendulumState(theta=theta, theta_dot=0.0, t=0.0, done=False)


def reward_fn(state: PendulumState, action: float) -> float:
    """Compute reward based on current state.

    Currently returns cos(theta), which rewards upright angles.

    Args:
        state (PendulumState): Current pendulum state.
        action (float): Applied torque (unused in current reward).

    Returns:
        float: Scalar reward value.
    """
    return np.cos(state.theta)

def step_pendulum_env(state: PendulumState, action: float, config: PendulumConfig):
    """Advance environment one step and return new state and reward.

    Args:
        state (PendulumState): Current state.
        action (float): Torque to apply.
        config (PendulumConfig): Environment configuration.

    Returns:
        tuple[PendulumState, float]: Next state and corresponding reward.
    """
    new_state = pendulum_dynamics(state, action, config)
    reward = reward_fn(new_state, action)
    return new_state, reward
