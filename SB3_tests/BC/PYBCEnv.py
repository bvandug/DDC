import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class BuckConverterEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, 
                 dt: float = 5e-6, 
                 max_episode_steps: int = 2000,
                 grace_period_steps: int = 50,
                 frame_skip: int = 10,
                 render_mode: str = None,
                 use_randomized_goal: bool = True,
                 fixed_goal_voltage: float = 30.0,
                 target_voltage_min: float = 28.5,
                 target_voltage_max: float = 31.5):
        super(BuckConverterEnv, self).__init__()
        
        # Core simulation parameters
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.grace_period_steps = grace_period_steps

        # Store the goal-setting strategy
        self.use_randomized_goal = use_randomized_goal
        self.fixed_goal_voltage = fixed_goal_voltage
        self.target_voltage_min = target_voltage_min
        self.target_voltage_max = target_voltage_max
        self.goal = 30.0  # Initial placeholder

        # Physical parameters of a buck converter
        self.V_in = 48.0 # Source voltage
        self.L = 100e-6 # Inductor
        self.C = 1000e-6 # Capacitor
        self.R = 10.0 # Resistor
        self.R_mosfet = 0.1 # MOSFET Resistance
        self.Vf_diode = 0.8 # Diode
        self.R_diode = 0.001 # Diode resistance
        self.f_sw = 10e3  # Switching frequency
        self.T_sw = 1 / self.f_sw # Switching period

        # Define action and observation spaces
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        
        # Observation: [voltage, error, derivative_error, goal]
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        # State variables
        self.v_out = 0.0
        self.i_L = 0.0
        self.prev_error = 0.0
        self.current_step = 0
        self.total_sim_time = 0.0

        # Rendering
        self.render_mode = render_mode
        self._plot_data = {}
        if self.render_mode == 'human':
            self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        self.fig, (self.ax_voltage, self.ax_duty) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        self.fig.suptitle('Python Buck Converter Control (48V Source) - Switching Model')

        self.line_voltage, = self.ax_voltage.plot([], [], 'b-', label="Actual Voltage", linewidth=2)
        self.line_goal, = self.ax_voltage.plot([], [], 'r--', label="Target Voltage")
        self.ax_voltage.set_ylabel("Voltage (V)")
        self.ax_voltage.legend(loc='best')
        self.ax_voltage.grid(True)
        self.ax_voltage.set_ylim(0, 55)

        self.line_duty, = self.ax_duty.plot([], [], 'm-', label="Duty Cycle (Action)")
        self.ax_duty.set_xlabel("Time (s)")
        self.ax_duty.set_ylabel("Duty Cycle")
        self.ax_duty.set_ylim(0, 1)
        self.ax_duty.legend(loc='best')
        self.ax_duty.grid(True)

    def _get_obs(self):
        error = self.v_out - self.goal
        step_duration = self.dt * self.frame_skip
        derivative_error = (error - self.prev_error) / step_duration if step_duration > 0 else 0.0
        return np.array([self.v_out, error, derivative_error, self.goal], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset state variables
        self.v_out = 0.0
        self.i_L = 0.0
        self.current_step = 0
        self.total_sim_time = 0.0
        
        # Set the goal for the new episode
        if self.use_randomized_goal:
            self.goal = self.np_random.uniform(self.target_voltage_min, self.target_voltage_max)
        else:
            self.goal = self.fixed_goal_voltage
            
        self.prev_error = self.v_out - self.goal

        # Reset plotting data
        if self.render_mode == 'human':
            self._plot_data = {'times': [], 'voltages': [], 'goals': [], 'duties': []}

        return self._get_obs(), {"goal": self.goal}

    def step(self, action):
        # Get the duty cycle and clip it to the allowed range
        duty_cycle = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        # Simulate the dynamics of the buck converter
        for _ in range(self.frame_skip):
            # Determine the exact time within the current ON/OFF switching period
            t_in_period = self.total_sim_time % self.T_sw
            
            # Check if the switch (MOSFET) is currently in its "ON" state
            if t_in_period < (duty_cycle * self.T_sw):
                # Switch is ON: Calculate the rate of change of inductor current (di/dt).
                # This is based on Kirchhoff's Voltage Law (KVL) for the primary loop:
                # V_inductor = V_input - V_mosfet_drop - V_output
                di_L_dt = (self.V_in - self.i_L * self.R_mosfet - self.v_out) / self.L
            else:
                # Switch is OFF: Calculate the rate of change of inductor current (di/dt).
                # The current now flows through the flywheel diode.
                # KVL for the secondary loop: V_inductor = -V_diode_drop - V_output
                di_L_dt = (-self.Vf_diode - self.i_L * self.R_diode - self.v_out) / self.L

            # Calculate the rate of change of the output capacitor voltage (dv/dt).
            # This is based on Kirchhoff's Current Law (KCL) at the output node:
            # I_capacitor = I_inductor - I_load_resistor
            dv_out_dt = (self.i_L - self.v_out / self.R) / self.C

            # Update the inductor current based on its rate of change over the small time step 'dt'.
            self.i_L += di_L_dt * self.dt

            # Enforce a physical constraint: the diode prevents current from flowing backward.
            if self.i_L < 0: self.i_L = 0
            
            # Update the output voltage based on its rate of change.
            self.v_out += dv_out_dt * self.dt

            # Advance total simulation time
            self.total_sim_time += self.dt

        self.current_step += 1
        
        obs = self._get_obs()
        error = obs[1]
        derivative_error = obs[2]

        reward = 1.0 / (1.0 + error**2)
        reward -= 0.01
        terminated = False
        if self.current_step > self.grace_period_steps:
            if not (0 < self.v_out < 53.0):
                reward -= 50.0
                terminated = True
        
        truncated = bool(self.current_step >= self.max_episode_steps)
        
        self.prev_error = error

        if self.render_mode == 'human':
            self._plot_data['times'].append(self.total_sim_time)
            self._plot_data['voltages'].append(self.v_out)
            self._plot_data['goals'].append(self.goal)
            self._plot_data['duties'].append(duty_cycle)
            self.render()
            
        return obs, reward, terminated, truncated, {"goal": self.goal}

    def render(self):
        if self.render_mode != 'human':
            return
        
        self.line_voltage.set_data(self._plot_data['times'], self._plot_data['voltages'])
        self.line_goal.set_data(self._plot_data['times'], self._plot_data['goals'])
        self.line_duty.set_data(self._plot_data['times'], self._plot_data['duties'])

        for ax in (self.ax_voltage, self.ax_duty):
            ax.relim()
            ax.autoscale_view(True, True, True)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.render_mode == 'human':
            plt.ioff()
            plt.close(self.fig)
