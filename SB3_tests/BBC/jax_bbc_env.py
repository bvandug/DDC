import gymnasium as gym
from gymnasium import spaces
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental.ode import odeint
import numpy as np

class JAXBuckBoostConverterEnv(gym.Env):
    def __init__(self, dt=5e-6, max_episode_time=0.03, target_voltage=-30.0):
        super().__init__()

        # --- Updated Circuit Parameters from Simulink ---
        self.Vin = 48.0
        self.L = 220e-6         # H
        self.C = 100e-6         # F
        self.R = 5.1            # Ohms
        self.Ron_switch = 0.1   # Ohms (MOSFET)
        self.Ron_diode = 0.01   # Ohms (Diode)
        self.Vf_diode = 0.0     # V (Diode forward voltage)

        self.fsw = 10e3
        self.Tsw = 1 / self.fsw

        self.dt = dt
        self.max_episode_time = max_episode_time
        self.target_voltage = target_voltage

        # --- Initial State: [iL, vC] ---
        self.time = 0.0
        self.state = jnp.array([0.0, 0.0])
        self.prev_state = jnp.array([0.0, 0.0])

        # --- Safety Limits ---
        self.I_L_MAX = 20.0  # Maximum inductor current (A)
        self.V_OUT_MAX = abs(target_voltage) * 1.5  # 150% of target
        self.V_OUT_MIN = abs(target_voltage) * 0.1   # 10% of target

        # --- Reward weights (adjusted for better balance) ---
        self.w_voltage = 1.0     # Voltage tracking (reduced to prevent domination)
        self.w_efficiency = 0.5  # Efficiency
        self.w_stability = 0.5   # Stability
        self.w_constraint = 2.0  # Constraint penalties (increased for safety)

        # --- Gym spaces ---
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        self.prev_error = 0.0
        self.prev_duty = 0.0  # initialize previous duty for reward

    @staticmethod
    @jit
    def pwm(t, D, Tsw):
        return jnp.where((t % Tsw) < D * Tsw, 1.0, 0.0)

    @staticmethod
    @jit
    def _jitted_reward(prev_duty, duty, V_out, V_ref):
        # Normalized voltage error
        e_norm = (V_ref - V_out) / V_ref
        # Voltage-error penalty
        r_error = -20.0 * jnp.abs(e_norm)
        # Smoothness penalty on duty changes
        r_smooth = -5.0 * (duty - prev_duty) ** 2
        # Tight-band bonus
        bonus = jnp.where(jnp.abs(e_norm) < 0.02, 1.0, 0.0)
        return r_error + r_smooth + bonus

    def buck_boost_dynamics(self, x, t, D):
        iL, vC = x
        u = self.pwm(t, D, self.Tsw)

        # When switch is ON: inductor charges from Vin, output isolated
        def on_state(_):
            diL = (self.Vin - iL * self.Ron_switch) / self.L
            dvC = -vC / (self.R * self.C)
            return jnp.array([diL, dvC])

        # When switch is OFF: inductor discharges through diode to output
        def off_state(_):
            diL = (-vC - iL * self.Ron_diode - self.Vf_diode) / self.L
            dvC = (iL - vC / self.R) / self.C
            return jnp.array([diL, dvC])

        return lax.cond(u == 1.0, on_state, off_state, operand=None)

    # ... (keep other helper methods unchanged) ...

    def calculate_reward(self, state, action, next_state):
        # Original reward method (unused by step)
        V_ref     = next_state["V_ref"]
        V_out     = next_state["V_out"]
        duty      = action
        prev_duty = getattr(self, "prev_duty", duty)

        e_norm = (V_ref - V_out) / V_ref
        K_error = 20.0
        r_error = -K_error * jnp.abs(e_norm)
        k_smooth = 5.0
        r_smooth = -k_smooth * (duty - prev_duty)**2
        epsilon = 0.02
        bonus   = jnp.where(jnp.abs(e_norm) < epsilon, 1.0, 0.0)
        reward = r_error + r_smooth + bonus

        self.prev_duty = duty
        return reward

    def step(self, action):
        duty = float(np.clip(action[0], 0.1, 0.9))
        self.prev_state = self.state.copy()

        t_span = jnp.linspace(self.time, self.time + self.dt, 2)
        self.state = odeint(self.buck_boost_dynamics, self.state, t_span, duty)[-1]

        self.time += self.dt
        vC = float(self.state[1])
        error = vC - self.target_voltage
        d_error = (error - self.prev_error) / self.dt

        # Use JIT-compiled reward for speed
        reward = float(
            self._jitted_reward(
                self.prev_duty,
                duty,
                vC,
                self.target_voltage
            )
        )
        # update prev_duty after computing reward
        self.prev_duty = duty

        terminated = self.time >= self.max_episode_time
        if abs(self.state[0]) > self.I_L_MAX * 1.2:
            terminated = True
            reward -= 100.0

        truncated = False
        self.prev_error = error

        observation = np.array([vC, error, d_error, self.target_voltage], dtype=np.float32)
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = jnp.array([0.0, 0.0])
        self.prev_state = jnp.array([0.0, 0.0])
        self.time = 0.0
        self.prev_error = -self.target_voltage
        self.prev_duty = 0.0
        observation = np.array([0.0, -self.target_voltage, 0.0, self.target_voltage], dtype=np.float32)
        return observation, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
