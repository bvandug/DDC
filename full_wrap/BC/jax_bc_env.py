import gym
from gym import spaces
import jax.numpy as jnp
from jax import jit
from jax import lax
from jax.experimental.ode import odeint
import numpy as np

class JAXBuckConverterEnv(gym.Env):
    def __init__(self, dt=5e-6, max_episode_time=0.03, target_voltage=30.0):
        super().__init__()

        # Simulation parameters
        self.Vin = 48.0
        self.L = 100e-6
        self.C = 1000e-6
        self.R = 10.0
        self.Ron_switch = 0.1
        self.Ron_diode = 0.001
        self.Vf_diode = 0.8

        self.fsw = 10e3
        self.Tsw = 1 / self.fsw

        self.dt = dt
        self.max_episode_time = max_episode_time
        self.target_voltage = target_voltage
        self.time = 0.0
        self.state = jnp.array([0.0, 0.0])  # iL, vC

        # Gym spaces
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        self.prev_error = 0.0

    @staticmethod
    @jit
    def pwm(t, D, Tsw):
        return jnp.where((t % Tsw) < D * Tsw, 1.0, 0.0)

    def buck_dynamics(self, x, t, D):
        iL, vC = x
        u = self.pwm(t, D, self.Tsw)

        vL = lax.cond(
            u == 1.0,
            lambda _: self.Vin - vC - iL * self.Ron_switch,
            lambda _: -vC - iL * self.Ron_diode - self.Vf_diode,
            operand=None
        )

        diL = vL / self.L
        dvC = (iL - vC / self.R) / self.C
        return jnp.array([diL, dvC])

    def step(self, action):
        duty = float(np.clip(action[0], 0.1, 0.9))

        t = jnp.linspace(self.time, self.time + self.dt, 2)
        self.state = odeint(self.buck_dynamics, self.state, t, duty)[-1]

        self.time += self.dt
        vC = float(self.state[1])
        error = vC - self.target_voltage
        d_error = (error - self.prev_error) / self.dt

        reward = 1.0 / (1.0 + error**2)
        terminated = self.time >= self.max_episode_time
        truncated = False

        self.prev_error = error

        observation = np.array([vC, error, d_error, self.target_voltage], dtype=np.float32)
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = jnp.array([0.0, 0.0])
        self.time = 0.0
        self.prev_error = -self.target_voltage
        observation = np.array([0.0, -self.target_voltage, 0.0, self.target_voltage], dtype=np.float32)
        return observation, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
