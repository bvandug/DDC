import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

# Environment configuration structure
class PendulumConfig(NamedTuple):
    m: float = 0.2        # pendulum mass (kg)
    L: float = 0.15       # pendulum length to COM (m)
    g: float = 9.8        # gravitational acceleration (m/s^2)
    dt: float = 0.01      # time step (s)
    angle_threshold: float = jnp.pi / 2
    max_torque: float = 2.0      # renamed from max_force
    max_episode_time: float = 5.0  # seconds

class PendulumState(NamedTuple):
    theta: float
    theta_dot: float
    t: float
    done: bool

def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

@jax.jit
def pendulum_dynamics(state: PendulumState, action: float, config: PendulumConfig) -> PendulumState:
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
    theta = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    theta = jnp.where(jnp.abs(theta) < 0.05, theta + 0.1, theta)
    return PendulumState(theta=theta, theta_dot=0.0, t=0.0, done=False)

def reward_fn(state: PendulumState, action: float, config: PendulumConfig) -> float:
    """Calculates reward with STRONGER penalties for velocity and effort."""
    position_reward = jnp.cos(state.theta)
    return position_reward

def step_pendulum_env(state: PendulumState, action: float, config: PendulumConfig):
    """Steps the environment forward using the shaped reward."""
    new_state = pendulum_dynamics(state, action, config)
    reward = reward_fn(new_state, action, config)
    return new_state, reward
