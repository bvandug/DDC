import gymnasium as gym
from gymnasium import spaces
import numpy as np

class JAXBuckBoostConverterEnv(gym.Env):
    def __init__(
        self,
        dt: float = 5e-6,
        max_episode_steps: int = 2000,
        frame_skip: int = 10,
        grace_period_steps: int = 50,
        target_voltage: float = -30.0
    ):
        super().__init__()

        # --- Circuit Parameters ---
        self.Vin = 48.0
        self.L = 220e-6         # H
        self.C = 100e-6         # F
        self.R = 5.1            # Ω
        self.Ron_switch = 0.1   # Ω (MOSFET)
        self.Ron_diode = 0.01   # Ω (Diode)
        self.Vf_diode = 0.0     # V (Diode forward drop)

        self.fsw = 10e3
        self.Tsw = 1 / self.fsw

        # Simulation parameters
        self.dt = dt
        self.frame_skip = frame_skip
        self.grace_period_steps = grace_period_steps
        self.max_episode_steps = max_episode_steps
        self.target_voltage = target_voltage

        # --- State ---
        self.time = 0.0
        self.state = np.zeros(2, dtype=float)   # [iL, vC]
        self.prev_state = self.state.copy()
        self.current_step = 0
        self.prev_error = 0.0
        self.prev_duty = 0.0

        # --- Safety Limits ---
        self.I_L_MAX = 20.0  # A
        self.V_OUT_MAX = abs(target_voltage) * 1.5
        self.V_OUT_MIN = abs(target_voltage) * 0.1

        # --- Gym spaces ---
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

    @staticmethod
    def pwm(t, D, Tsw):
        return 1.0 if (t % Tsw) < D * Tsw else 0.0

    def buck_boost_dynamics(self, x, duty):
        iL, vC = x
        u = self.pwm(self.time, duty, self.Tsw)
        if u == 1.0:
            diL = (self.Vin - iL * self.Ron_switch) / self.L
            dvC = -vC / (self.R * self.C)
        else:
            diL = (-vC - iL * self.Ron_diode - self.Vf_diode) / self.L
            dvC = (iL - vC / self.R) / self.C
        return np.array([diL, dvC], dtype=float)

    def calculate_reward(self, state, duty, next_vC):
        e_norm = (self.target_voltage - next_vC) / self.target_voltage
        r_error = -20.0 * abs(e_norm)
        r_smooth = -5.0 * (duty - self.prev_duty) ** 2
        bonus   = 1.0 if abs(e_norm) < 0.02 else 0.0
        return r_error + r_smooth + bonus

    def step(self, action):
        duty = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        # integrate for frame_skip sub-steps
        for _ in range(self.frame_skip):
            self.prev_state = self.state.copy()
            prev_err = self.prev_error

            deriv = self.buck_boost_dynamics(self.state, duty)
            self.state += deriv * self.dt
            self.time += self.dt

        self.current_step += 1

        # observations
        vC = float(self.state[1])
        error = vC - self.target_voltage
        d_error = (error - prev_err) / (self.dt * self.frame_skip)

        # reward
        reward = self.calculate_reward(self.prev_state, duty, vC)
        self.prev_duty = duty

        # post-grace penalties
        if self.current_step > self.grace_period_steps:
            if abs(self.state[0]) > self.I_L_MAX:
                reward -= 50.0
            if not (self.V_OUT_MIN < abs(vC) < self.V_OUT_MAX):
                reward -= 50.0

        terminated = self.current_step >= self.max_episode_steps
        truncated = False
        self.prev_error = error

        obs = np.array([vC, error, d_error, self.target_voltage], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0.0
        self.state = np.zeros(2, dtype=float)
        self.prev_state = self.state.copy()
        self.current_step = 0
        self.prev_error = 0.0
        self.prev_duty = 0.0
        obs = np.array([0.0, -self.target_voltage, 0.0, self.target_voltage], dtype=np.float32)
        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
