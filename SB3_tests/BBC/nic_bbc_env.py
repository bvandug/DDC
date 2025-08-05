import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class BuckBoostConverterEnv(gym.Env):
    """
    Gymnasium environment for a DC-DC Buck-Boost Converter. 🚀

    This environment simulates an inverting buck-boost converter. The agent's goal is to
    control the duty cycle of a PWM signal to maintain a target output voltage magnitude.

    It uses a multi-part reward function to penalize error, overshoot, and
    instability while rewarding high-precision control.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 dt: float = 5e-6,
                 # MODIFIED: Timesteps adjusted for a 0.1s simulation
                 max_episode_steps: int = 2000,
                 grace_period_steps: int = 1000,
                 frame_skip: int = 10,
                 render_mode: str = None,
                 render_frequency: int = 1,
                 use_randomized_goal: bool = False,
                 fixed_goal_voltage: float = 30.0,
                 target_voltage_min: float = 30.0,
                 target_voltage_max: float = 110.0):
        super(BuckBoostConverterEnv, self).__init__()

        # Core simulation parameters
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.grace_period_steps = grace_period_steps
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.render_frequency = render_frequency

        # Goal-setting strategy
        self.use_randomized_goal = use_randomized_goal
        self.fixed_goal_voltage = fixed_goal_voltage
        self.target_voltage_min = target_voltage_min
        self.target_voltage_max = target_voltage_max
        self.goal = 30.0

        # Physical Parameters from ddc_matt_report.pdf
        self.V_in = 48.0
        self.L = 220e-6
        self.C = 8200e-6
        self.R = 5.1
        self.R_mosfet = 0.1
        self.Vf_diode = 0.8
        self.R_diode = 0.001
        self.f_sw = 7.8e3
        self.T_sw = 1 / self.f_sw
        self.TOLERANCE_VOLTS = 0.5

        # Action and observation spaces
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        # State variables
        self.v_out, self.i_L, self.prev_error = 0.0, 0.0, 0.0
        self.current_step, self.total_sim_time = 0, 0.0

        # Rendering
        self._plot_data = {}
        if self.render_mode == 'human':
            self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        self.fig, (self.ax_voltage, self.ax_duty) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        self.fig.suptitle('Live Training View: Buck-Boost Converter')
        self.line_voltage, = self.ax_voltage.plot([], [], 'b-', label="Actual Voltage", linewidth=2)
        self.line_goal, = self.ax_voltage.plot([], [], 'r--', label="Target Voltage")
        self.ax_voltage.set_ylabel("Voltage (V)"), self.ax_voltage.legend(loc='best'), self.ax_voltage.grid(True)
        self.ax_voltage.set_ylim(-120, 5)
        self.line_duty, = self.ax_duty.plot([], [], 'm-', label="Duty Cycle")
        self.ax_duty.set_xlabel("Time (s)"), self.ax_duty.set_ylabel("Duty Cycle"), self.ax_duty.legend(loc='best')
        self.ax_duty.set_ylim(0, 1), self.ax_duty.grid(True)

    def _get_obs(self):
        error = self.goal - abs(self.v_out)
        step_duration = self.dt * self.frame_skip
        derivative_error = (error - self.prev_error) / step_duration if step_duration > 0 else 0.0
        self.prev_error = error
        return np.array([self.v_out, error, derivative_error, self.goal], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.v_out, self.i_L, self.current_step, self.total_sim_time = 0.0, 0.0, 0, 0.0
        if self.use_randomized_goal:
            self.goal = self.np_random.uniform(self.target_voltage_min, self.target_voltage_max)
        else: self.goal = self.fixed_goal_voltage
        self.prev_error = self.goal - abs(self.v_out)
        if self.render_mode == 'human':
            self._plot_data = {'times': [], 'voltages': [], 'goals': [], 'duties': []}
        return self._get_obs(), {"goal": self.goal}

    def step(self, action):
        duty_cycle = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        for _ in range(self.frame_skip):
            t_in_period = self.total_sim_time % self.T_sw
            if t_in_period < (duty_cycle * self.T_sw):
                di_L_dt = (self.V_in - self.i_L * self.R_mosfet) / self.L
                dv_out_dt = -self.v_out / (self.R * self.C)
            else:
                di_L_dt = (-self.Vf_diode - self.i_L * self.R_diode - self.v_out) / self.L
                dv_out_dt = (-self.i_L - self.v_out / self.R) / self.C
            self.i_L = max(0, self.i_L + di_L_dt * self.dt)
            self.v_out += dv_out_dt * self.dt
            self.total_sim_time += self.dt
        self.current_step += 1
        obs = self._get_obs()

        voltage_error = self.v_out - (-self.goal)
        err_ratio = voltage_error / self.goal
        reward = -(voltage_error ** 2)
        if abs(voltage_error) < self.TOLERANCE_VOLTS: reward += 0.5
        if abs(voltage_error) > 0.05 * self.goal: reward -= 0.2 * (abs(voltage_error) / self.goal - 0.05)

        terminated, truncated = False, bool(self.current_step >= self.max_episode_steps)
        if self.current_step > self.grace_period_steps:
            upper_bound = -0.1 * self.goal
            lower_bound = -1.5 * self.goal
            # if not (lower_bound < self.v_out < upper_bound):
            #     reward, terminated = -100.0, True

        if self.render_mode == 'human' and self.current_step % self.render_frequency == 0:
            self._plot_data['times'].append(self.total_sim_time)
            self._plot_data['voltages'].append(self.v_out)
            self._plot_data['goals'].append(-self.goal)
            self._plot_data['duties'].append(duty_cycle)
            self.render()

        return obs, reward, terminated, truncated, {"goal": self.goal}

    def render(self):
        if self.render_mode != 'human': return
        self.line_voltage.set_data(self._plot_data['times'], self._plot_data['voltages'])
        self.line_goal.set_data(self._plot_data['times'], self._plot_data['goals'])
        self.line_duty.set_data(self._plot_data['times'], self._plot_data['duties'])
        for ax in (self.ax_voltage, self.ax_duty):
            ax.relim(), ax.autoscale_view(True, True, True)
        self.fig.canvas.draw(), self.fig.canvas.flush_events()

    def close(self):
        if self.render_mode == 'human':
            plt.ioff(), plt.close(self.fig)