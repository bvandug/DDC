# ip_numpy.py (NumPy-based dynamics for SB3)
import numpy as np
from typing import NamedTuple

class PendulumConfig(NamedTuple):
    m: float = 0.2
    M: float = 0.5
    L: float = 0.15
    g: float = 9.8
    dt: float = 0.01
    angle_threshold: float = np.pi / 2
    max_force: float = 10.0
    max_episode_time: float = 5.0

class PendulumState(NamedTuple):
    theta: float
    theta_dot: float
    t: float
    done: bool

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def pendulum_dynamics(state: PendulumState, action: float, config: PendulumConfig) -> PendulumState:
    u = np.clip(action, -config.max_force, config.max_force)

    theta = state.theta
    theta_dot = state.theta_dot
    m, M, l, g = config.m, config.M, config.L, config.g

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    denom = m * l * cos_theta**2 - (M + m) * l
    denom = denom if abs(denom) >= 1e-5 else np.sign(denom) * 1e-5

    theta_ddot = (
        u * cos_theta
        - (M + m) * g * sin_theta
        + m * l * theta_dot**2 * cos_theta * sin_theta
    ) / denom

    theta_dot_new = theta_dot + theta_ddot * config.dt
    theta_new = theta + theta_dot_new * config.dt
    t_new = state.t + config.dt

    done = abs(theta_new) > config.angle_threshold or t_new >= config.max_episode_time

    return PendulumState(
        theta=angle_normalize(theta_new),
        theta_dot=theta_dot_new,
        t=t_new,
        done=done
    )

def reset_pendulum_env(seed, config: PendulumConfig) -> PendulumState:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-1.0, 1.0)
    if abs(theta) < 0.05:
        theta += 0.1
    return PendulumState(theta=theta, theta_dot=0.0, t=0.0, done=False)

def reward_fn(state: PendulumState, action: float) -> float:
    return np.cos(state.theta)

def step_pendulum_env(state: PendulumState, action: float, config: PendulumConfig):
    new_state = pendulum_dynamics(state, action, config)
    reward = reward_fn(new_state, action)
    return new_state, reward
