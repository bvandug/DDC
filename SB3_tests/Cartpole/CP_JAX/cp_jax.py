import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

# Environment configuration
class CartPoleConfig(NamedTuple):
    m: float = 0.2     # pole mass
    M: float = 0.5     # cart mass
    L: float = 0.15    # pendulum length to COM
    g: float = 9.8     # gravity
    dt: float = 0.01   # timestep
    angle_threshold: float = jnp.pi / 2  # ±90 deg
    max_force: float = 10.0
    max_episode_time: float = 5.0  # seconds

# Full state including cart pos/vel for render
class CartPoleState(NamedTuple):
    x: float
    x_dot: float
    theta: float
    theta_dot: float
    t: float
    done: bool

def angle_normalize(x):
    """Normalize angle(s) to the range (-π, π].

        Parameters
        ----------
        x : float | jnp.ndarray
            Angle in radians; can be scalar or array-like.

        Returns
        -------
        jnp.ndarray
            Angle(s) wrapped to (-π, π].
    """

    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

@jax.jit
def cartpole_dynamics(state: CartPoleState, action: float, config: CartPoleConfig) -> CartPoleState:
    """Advance the cart-pole dynamics by one Euler step (JAX, pure, jit-able).

        Applies force saturation, integrates cart position/velocity and pole
        angle/angular velocity using simple Euler integration, advances time,
        and flags termination if the angle exceeds the threshold or the episode
        time limit is reached. The returned angle is wrapped via
        `angle_normalize`.

        Parameters
        ----------
        state : CartPoleState
            Current system state.
        action : float
            Applied horizontal force (N), clipped to ±config.max_force.
        config : CartPoleConfig
            Physical parameters and integration settings.

        Returns
        -------
        CartPoleState
            Next state after one integration step.
    """

    u = jnp.clip(action, -config.max_force, config.max_force)

    x, x_dot = state.x, state.x_dot
    theta, theta_dot = state.theta, state.theta_dot
    m, M, l, g, dt = config.m, config.M, config.L, config.g, config.dt

    total_mass = m + M
    polemass_length = m * l
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)

    temp = (u + polemass_length * theta_dot**2 * sintheta) / total_mass

    theta_acc = (g * sintheta - costheta * temp) / (
        l * (4.0 / 3.0 - m * costheta**2 / total_mass)
    )

    x_acc = temp - (polemass_length * theta_acc * costheta) / total_mass

    # Euler integration
    x_dot_new = x_dot + x_acc * dt
    x_new     = x + x_dot_new * dt
    theta_dot_new = theta_dot + theta_acc * dt
    theta_new     = theta + theta_dot_new * dt
    t_new = state.t + dt

    done = jnp.logical_or(
        jnp.abs(theta_new) > config.angle_threshold,
        t_new >= config.max_episode_time
    )

    return CartPoleState(
        x=x_new,
        x_dot=x_dot_new,
        theta=angle_normalize(theta_new),
        theta_dot=theta_dot_new,
        t=t_new,
        done=done
    )

@jax.jit
def reset_cartpole_env(key, config: CartPoleConfig) -> CartPoleState:
    """ Create an initial cart-pole state with a small randomized angle.

        Samples θ ~ U[-0.1, 0.1] rad; if |θ| < 0.05, adds +0.1 to avoid
        starting too close to perfectly upright. Other state components are
        initialized to zero and the time is set to 0.

        Parameters
        ----------
        key
            JAX PRNG key.
        config : CartPoleConfig
            Environment configuration.

        Returns
        -------
        CartPoleState
            Fresh initial state (done=False).
    """

    theta = jax.random.uniform(key, minval=-0.1, maxval=0.1)
    theta = jnp.where(jnp.abs(theta) < 0.05, theta + 0.1, theta)
    return CartPoleState(
        x=0.0,
        x_dot=0.0,
        theta=theta,
        theta_dot=0.0,
        t=0.0,
        done=False
    )

def reward_fn(state: CartPoleState, action: float) -> float:
    """Compute the immediate reward for a state-action pair.

        Uses uprightness as the objective: `cos(theta)`, which equals 1 when
        the pole is perfectly upright and decreases as it tilts.

        Parameters
        ----------
        state : CartPoleState
            Current state (theta is used).
        action : float
            Control input (unused here, kept for interface consistency).

        Returns
        -------
        float
            Reward value (higher is better).
    """

    return jnp.cos(state.theta)

def step_cartpole_env(state: CartPoleState, action: float, config: CartPoleConfig):
    """Apply one environment step: dynamics update + reward evaluation.

        Parameters
        ----------
        state : CartPoleState
            Current state.
        action : float
            Control input (N).
        config : CartPoleConfig
            Environment configuration.

        Returns
        -------
        tuple[CartPoleState, float]
            The next state and the scalar reward.
    """

    new_state = cartpole_dynamics(state, action, config)
    reward = reward_fn(new_state, action)
    return new_state, reward
