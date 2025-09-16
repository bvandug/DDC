"""Lightweight JAX pendulum environment.

Implements a minimal pendulum simulator with Euler integration,
JIT-compiled dynamics, and a simple reward. Suitable for RL
experiments that require fast, differentiable physics.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple



class PendulumConfig(NamedTuple):
    """Configuration parameters for the pendulum environment.

    Attributes:
        m (float): Pendulum mass in kilograms.
        L (float): Distance from pivot to center of mass in meters.
        g (float): Gravitational acceleration in m/s^2.
        dt (float): Simulation time step in seconds.
        angle_threshold (float): Episode terminates if |theta| exceeds this (rad).
        max_torque (float): Action torque is clipped to ±this value (N·m).
        max_episode_time (float): Hard time limit per episode in seconds.
    """
    m: float = 0.2        
    L: float = 0.15      
    g: float = 9.8        
    dt: float = 0.01      
    angle_threshold: float = jnp.pi / 2
    max_torque: float = 2.0      
    max_episode_time: float = 5.0  

class PendulumState(NamedTuple):
    """Container for the pendulum's instantaneous state.

    Attributes:
        theta (float): Angular position in radians (normalized later to [-π, π]).
        theta_dot (float): Angular velocity in radians per second.
        t (float): Elapsed simulation time in seconds.
        done (bool): True if the episode has terminated.
    """
    theta: float
    theta_dot: float
    t: float
    done: bool

def angle_normalize(x):
    """Normalize an angle to the range [-π, π].

    Args:
        x (float): Angle in radians.

    Returns:
        float: Angle wrapped into [-π, π] for numerical stability.
    """
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

@jax.jit
def pendulum_dynamics(state: PendulumState, action: float, config: PendulumConfig) -> PendulumState:

    """Compute one physics step of pendulum dynamics (Euler integration).

    The torque input is clipped to ±max_torque. The next state is computed
    via explicit Euler integration and the angle is normalized to [-π, π].

    Args:
        state (PendulumState): Current state (theta, theta_dot, time, done).
        action (float): Control input torque (N·m) before clipping.
        config (PendulumConfig): Environment constants and limits.

    Returns:
        PendulumState: Next state after one time step.

    Notes:
        This function is JIT-compiled with JAX for performance.
    """

    # Clip input to ±max_torque
    tau = jnp.clip(action, -config.max_torque, config.max_torque)

    theta     = state.theta
    theta_dot = state.theta_dot
    m, L, g   = config.m, config.L, config.g

    # Moment of inertia about pivot
    I = m * L**2

    # Pure‑pendulum acceleration
    theta_ddot = (-m * g * L * jnp.sin(theta) + tau) / I

    # Euler integration
    theta_dot_new = theta_dot + theta_ddot * config.dt
    theta_new     = theta + theta_dot_new * config.dt
    t_new         = state.t + config.dt

    # Termination: angle out of bounds or time up
    done = jnp.logical_or(
        jnp.abs(theta_new) > config.angle_threshold,
        t_new >= config.max_episode_time
    )

    return PendulumState(
        theta=angle_normalize(theta_new),
        theta_dot=theta_dot_new,
        t=t_new,
        done=done
    )

@jax.jit
def reset_pendulum_env(key, config: PendulumConfig) -> PendulumState:
    """Initialize a new episode with a randomized starting angle.

    The initial theta is sampled uniformly from [-1, 1] rad. Very small
    magnitudes are nudged away from 0 to avoid trivial starts.

    Args:
        key: JAX PRNGKey used to sample the initial angle.
        config (PendulumConfig): Environment configuration.

    Returns:
        PendulumState: Initial state with theta_dot=0, t=0, done=False.

    Notes:
        JIT-compiled for consistency and speed in vectorized setups.
    """

    theta = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    theta = jnp.where(jnp.abs(theta) < 0.05, theta + 0.1, theta)
    return PendulumState(theta=theta, theta_dot=0.0, t=0.0, done=False)

def reward_fn(state: PendulumState, action: float, config: PendulumConfig) -> float:
    """Compute the scalar reward for a given state-action pair.

    Currently returns cos(theta), which rewards the upright pose (theta=0).
    Commented lines show optional shaping terms for velocity and effort.

    Args:
        state (PendulumState): State at which the reward is evaluated.
        action (float): Applied torque (N·m).
        config (PendulumConfig): Environment configuration (unused by default).

    Returns:
        float: Reward value (higher is better).
    """
    position_reward = jnp.cos(state.theta)
    return position_reward

def step_pendulum_env(state: PendulumState, action: float, config: PendulumConfig):
    """Advance the environment by one step and compute the reward.

    Calls the dynamics to obtain the next state, then evaluates the reward
    on the new state (post-dynamics).

    Args:
        state (PendulumState): Current environment state.
        action (float): Control input torque (N·m).
        config (PendulumConfig): Environment configuration.

    Returns:
        tuple[PendulumState, float]: (next_state, reward) pair.
    """
    new_state = pendulum_dynamics(state, action, config)
    reward = reward_fn(new_state, action, config)
    return new_state, reward
