import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

# Environment configuration structure
class PendulumConfig(NamedTuple):
    m: float = 0.2     # pendulum mass (kg)
    M: float = 0.5     # cart mass (kg)
    L: float = 0.15    # pendulum length to COM (m)
    g: float = 9.8     # gravitational acceleration (m/s^2)
    dt: float = 0.01   # time step (s)
    angle_threshold: float = jnp.pi / 2
    max_force: float = 10.0

class PendulumState(NamedTuple):
    theta: float
    theta_dot: float
    t: float
    done: bool

def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

def pendulum_dynamics(state: PendulumState, action: float, config: PendulumConfig) -> PendulumState:
    u = jnp.clip(action, -config.max_force, config.max_force)

    theta = state.theta
    theta_dot = state.theta_dot
    m, M, l, g = config.m, config.M, config.L, config.g

    # Equations from Simulink model
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    denom = M + m - m * cos_theta**2

    theta_ddot = (
        u * cos_theta
        - (M + m) * g * sin_theta
        + m * l * theta_dot**2 * cos_theta * sin_theta
    ) / (m * l * cos_theta**2 - (M + m) * l)

    theta_dot_new = theta_dot + theta_ddot * config.dt
    theta_new = theta + theta_dot_new * config.dt

    t_new = state.t + config.dt

    done = jnp.abs(theta_new) > config.angle_threshold

    return PendulumState(
        theta=angle_normalize(theta_new),
        theta_dot=theta_dot_new,
        t=t_new,
        done=done
    )

def reset_pendulum_env(key, config: PendulumConfig) -> PendulumState:
    theta = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    theta = jnp.where(jnp.abs(theta) < 0.05, theta + 0.1, theta)
    return PendulumState(theta=theta, theta_dot=0.0, t=0.0, done=False)

def reward_fn(state: PendulumState, action: float) -> float:
    return jnp.cos(state.theta)

def step_pendulum_env(state: PendulumState, action: float, config: PendulumConfig):
    new_state = pendulum_dynamics(state, action, config)
    reward = reward_fn(new_state, action)
    return new_state, reward
