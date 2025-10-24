import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

# Use 64-bit floats to match Simulink doubles
jax.config.update('jax_enable_x64', True)

class PendulumConfig(NamedTuple):
    m: float = 0.2
    L: float = 0.15
    g: float = 9.8              # <-- match Simulink (9.8)
    dt: float = 0.01
    angle_threshold: float = jnp.pi / 2
    max_torque: float = 2.0
    max_episode_time: float = 5.0

class PendulumState(NamedTuple):
    theta: float       # UNWRAPPED internal angle (rad)
    theta_dot: float
    t: float
    done: bool

def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

# For exactness validation you can remove JIT
@jax.jit
def pendulum_dynamics(state: PendulumState, action: float, config: PendulumConfig) -> PendulumState:
    # If Simulink has no saturation on tau, remove this clip.
    tau = jnp.clip(action, -config.max_torque, config.max_torque)

    theta     = state.theta
    theta_dot = state.theta_dot
    m, L, g   = config.m, config.L, config.g

    # Moment of inertia about pivot: use L*L (not L**2) to mirror C as closely as possible
    I = m * L * L

    # EXACT grouping as in pendulum.c: ( -m*g*L*sin(theta) + tau ) / (m*L*L)
    num = - (m * g * L) * jnp.sin(theta) + tau
    theta_ddot = num / I

    # Forward Euler cascade (ode1): theta uses OLD omega; omega uses OLD accel
    theta_new     = theta     + theta_dot  * config.dt
    theta_dot_new = theta_dot + theta_ddot * config.dt

    t_new = state.t + config.dt

    # Termination on UNWRAPPED theta to match Simulink Stop block
    done = jnp.logical_or(
        jnp.abs(theta_new) > config.angle_threshold,
        t_new >= config.max_episode_time
    )

    # IMPORTANT: keep theta UNWRAPPED in the state; only wrap for observations if needed
    return PendulumState(theta=theta_new, theta_dot=theta_dot_new, t=t_new, done=done)

def reset_pendulum_env(key, config: PendulumConfig) -> PendulumState:
    # For exact comparisons, set the same ICs as the Simulink model uses.
    # If you need randomness, keep it—but identity tests should use fixed ICs.
    theta = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    theta = jnp.where(jnp.abs(theta) < 0.05, theta + 0.1, theta)
    return PendulumState(theta=theta, theta_dot=0.0, t=0.0, done=False)

def reward_fn(state: PendulumState, action: float, config: PendulumConfig) -> float:
    return jnp.cos(angle_normalize(state.theta))  # wrap only for reward/obs if you like

def step_pendulum_env(state: PendulumState, action: float, config: PendulumConfig):
    new_state = pendulum_dynamics(state, action, config)
    reward = reward_fn(new_state, action, config)
    return new_state, reward
