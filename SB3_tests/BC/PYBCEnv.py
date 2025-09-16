import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class BuckConverterEnv(gym.Env):
    """
    A Gymnasium environment for controlling a non-ideal DC-DC buck converter.

    The agent learns to control the duty cycle to regulate the output voltage.
    The state-space equations are a non-ideal implementation based on the
        ideal model described in: "Design and Analysis of Buck Converter"
        by M. Gupta and S. P. Phulambrikar, IJERT, 2014.

    Attributes:
        V_in (float): Source Voltage (V).
        L (float): Inductor Inductance (H).
        C (float): Capacitor Capacitance (F).
        R (float): Load Resistor Resistance (Ohm).
        action_space (gym.spaces.Box): The agent's action space (duty cycle).
        observation_space (gym.spaces.Box): The environment's observation space.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 dt: float = 5e-6,
                 max_episode_steps: int = 600,
                 grace_period_steps: int = 50,
                 frame_skip: int = 10,
                 render_mode: str = None,
                 use_randomized_goal: bool = True,
                 fixed_goal_voltage: float = 30.0,
                 target_voltage_min: float = 28.5,
                 target_voltage_max: float = 31.5,
                 voltage_noise_std: float = 0.0):
        """Initializes the Buck Converter environment.

        Args:
            dt (float): The time step for the internal physics simulation (s).
            max_episode_steps (int): The max number of agent steps per episode.
            grace_period_steps (int): Initial steps to ignore termination.
            frame_skip (int): The number of physics simulations per agent step.
            render_mode (str): The rendering mode for gymnasium compatibility (default to none).
            use_randomized_goal (bool): If True, randomise target voltage.
            fixed_goal_voltage (float): Target voltage if randomisation is off.
            target_voltage_min (float): Min value for randomised target voltage.
            target_voltage_max (float): Max value for randomised target voltage.
            voltage_noise_std (float): Std deviation of Gaussian sensor noise injected into voltage observations.
        """
        super(BuckConverterEnv, self).__init__()

        # Core simulation parameters
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.grace_period_steps = grace_period_steps

        # Goal-setting strategy
        self.use_randomized_goal = use_randomized_goal
        self.fixed_goal_voltage = fixed_goal_voltage
        self.target_voltage_min = target_voltage_min
        self.target_voltage_max = target_voltage_max
        self.goal = 30.0

        # Sensor noise parameters
        self.voltage_noise_std = voltage_noise_std

        # Physical parameters of a non-ideal buck converter
        self.V_in = 48.0
        self.L = 100e-6
        self.C = 1000e-6
        self.R = 10.0
        self.R_mosfet = 0.1 # Mosfet resistance
        self.Vf_diode = 0.8 # Diode forward voltage
        self.R_diode = 0.001 # Diode resistance
        self.f_sw = 10e3 # Switching frequency
        self.T_sw = 1 / self.f_sw # Switching period

        # Action: duty_cycle (continuous in the range [0.1, 0.9])
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)

        # Observation: [voltage, error, derivative_error, goal]
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        # Internal state variables
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
        """
        Constructs the observation vector from the current state.

        Returns:
            numpy.ndarray: The observation vector containing the noisy voltage, error, derivative of the error and the goal voltage.
        """
        noisy_v_out = self.v_out + self.np_random.normal(0, self.voltage_noise_std) # Add gaussian noise to voltage observation
        error = noisy_v_out - self.goal
        step_duration = self.dt * self.frame_skip
        derivative_error = (error - self.prev_error) / step_duration if step_duration > 0 else 0.0
        return np.array([noisy_v_out, error, derivative_error, self.goal], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.

        Returns:
            tuple: A tuple containing the initial observation and info dict (Gymnasium format).
        """
        super().reset(seed=seed)
        self.v_out, self.i_L, self.current_step, self.total_sim_time = 0.0, 0.0, 0, 0.0

        # Set the goal of the episode
        if self.use_randomized_goal:
            self.goal = self.np_random.uniform(self.target_voltage_min, self.target_voltage_max)
        else:
            self.goal = self.fixed_goal_voltage

        self.prev_error = self.v_out - self.goal
        if self.render_mode == 'human':
            self._plot_data = {'times': [], 'voltages': [], 'goals': [], 'duties': []}
        return self._get_obs(), {"goal": self.goal}

    def step(self, action):
        """
        Advances the simulation by one agent step.

        Args:
            action (np.ndarray): The action from the agent (duty cycle) taken from SB3.

        Returns:
            tuple: A tuple containing the new observation, reward, terminated flag, truncated flag, and an info dictionary.
        """
        duty_cycle = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        # Run the physics simulation for 'frame_skip' steps
        for _ in range(self.frame_skip):
            t_in_period = self.total_sim_time % self.T_sw
            if t_in_period < (duty_cycle * self.T_sw):
                # Switch ON state, extended with MOSFET resistance
                di_L_dt = (self.V_in - self.i_L * self.R_mosfet - self.v_out) / self.L
            else:
                # Switch OFF state, extended with diode voltage drop and resistance
                di_L_dt = (-self.Vf_diode - self.i_L * self.R_diode - self.v_out) / self.L

            # Capacitor dynamics are the same in both states
            dv_out_dt = (self.i_L - self.v_out / self.R) / self.C

            # Update state variables using forward Euler integration
            self.i_L += di_L_dt * self.dt
            if self.i_L < 0:
                self.i_L = 0
            self.v_out += dv_out_dt * self.dt
            self.total_sim_time += self.dt

        self.current_step += 1
        obs = self._get_obs()
        noisy_error = obs[1]

        # Reward is based on minimizing the true (non-noisy) squared error
        true_error = self.v_out - self.goal

        # Calculate reward
        reward = 1.0 / (1.0 + true_error**2) - 0.01

        # Terminate if voltage goes out of a safe range (after the grace period) and give a large negative reward
        terminated = False
        if self.current_step > self.grace_period_steps and not (0 < self.v_out < 53.0):
            reward -= 50.0
            terminated = True

        # Truncate if max steps reached (episode ends)
        truncated = self.current_step >= self.max_episode_steps
        self.prev_error = noisy_error

        info = {
            "goal": self.goal,
            "true_voltage": self.v_out,
            "true_error": true_error,
            "inductor_current": self.i_L,
            "duty_cycle": duty_cycle
        }

        if self.render_mode == 'human':
            self._plot_data['times'].append(self.total_sim_time)
            self._plot_data['voltages'].append(self.v_out)
            self._plot_data['goals'].append(self.goal)
            self._plot_data['duties'].append(duty_cycle)
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Updates the plot for human rendering.
        """
        if self.render_mode != 'human': return
        self.line_voltage.set_data(self._plot_data['times'], self._plot_data['voltages'])
        self.line_goal.set_data(self._plot_data['times'], self._plot_data['goals'])
        self.line_duty.set_data(self._plot_data['times'], self._plot_data['duties'])
        for ax in (self.ax_voltage, self.ax_duty):
            ax.relim()
            ax.autoscale_view(True, True, True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """
        Closes the rendering window.
        """
        if self.render_mode == 'human':
            plt.ioff()
            plt.close(self.fig)