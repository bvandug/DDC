import gym
from gym import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

class BCSimulinkEnv(gym.Env):
    """
    Robust environment for controlling a Buck Converter in Simulink.
    This version includes enhanced plotting with tolerance bands and y-axis zoom control.
    """
    def __init__(self, model_name="bcSim", dt=5e-6, max_episode_time=0.006,
                 grace_period_steps=50,
                 frame_skip=10,
                 enable_plotting=False,
                 target_voltage=30.0,
                 voltage_plot_ylim=None): # New parameter for zooming
        
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()
        print(f"Loading {model_name}...")
        self.eng.load_system(model_name, nargout=0)
        self.eng.set_param(model_name, 'FastRestart', 'on', nargout=0)

        # Initializing the environment parameters
        self.model_name = model_name
        self.dt = dt
        self.max_episode_time = max_episode_time
        self.frame_skip = frame_skip
        self.enable_plotting = enable_plotting
        self.grace_period_steps = grace_period_steps
        self.goal = target_voltage
        self.voltage_plot_ylim = voltage_plot_ylim # Store the y-limit
        self.steps_taken = 0
        self.current_time = 0.0

        # Defining the action and observation spaces
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        self.prev_error = 0
        self._np_random = None

        if self.enable_plotting:
            self._setup_plot()

    def _setup_plot(self):
        """
        Sets up the live plot, including tolerance bands and custom y-limits.
        """
        print("Setting up the live plot...")
        plt.ion()
        self.fig, (self.ax_voltage, self.ax_duty) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        self.fig.suptitle('BC Simulink Control (48V Source Voltage)')
        
        # Voltage Plot
        self.line_voltage, = self.ax_voltage.plot([], [], 'b-', label="Actual Voltage", linewidth=2)
        self.line_goal,    = self.ax_voltage.plot([], [], 'r--', label="Target Voltage")
        self.line_plus_0_5v, = self.ax_voltage.plot([], [], 'g:', label="±0.5V Tolerance")
        self.line_minus_0_5v, = self.ax_voltage.plot([], [], 'g:')
        self.line_plus_1v, = self.ax_voltage.plot([], [], 'k:', label="±1.0V Tolerance")
        self.line_minus_1v, = self.ax_voltage.plot([], [], 'k:')
        
        self.ax_voltage.set_ylabel("Voltage (V)")
        self.ax_voltage.legend(loc='best')
        self.ax_voltage.grid(True)
        # Set y-axis limits if provided
        if self.voltage_plot_ylim:
            self.ax_voltage.set_ylim(self.voltage_plot_ylim)
        
        # Duty Cycle Plot
        self.line_duty,    = self.ax_duty.plot([], [], 'm-', label="Duty Cycle (Action)")
        self.ax_duty.set_xlabel("Time (s)")
        self.ax_duty.set_ylabel("Duty Cycle")
        self.ax_duty.set_ylim(0, 1)
        self.ax_duty.legend(loc='best')
        self.ax_duty.grid(True)
        
        # Data storage lists
        self._times, self._voltages, self._goals, self._duties = [], [], [], []
        self._plus_0_5v, self._minus_0_5v = [], []
        self._plus_1v, self._minus_1v = [], []


    def get_data(self):
        """Gets the data from the simulation."""
        voltage_out = self.eng.eval("out.voltage", nargout=1)
        time_out = self.eng.eval("out.tout", nargout=1)
        final_voltage = voltage_out[-1][0] if not isinstance(voltage_out, float) else voltage_out
        final_time = time_out[-1][0] if not isinstance(time_out, float) else time_out
        return final_voltage, final_time

    def reset(self, seed=None, options=None):
        """Resets the environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            
        print("\n--- Episode Reset ---")
        self.current_time = 0.0
        self.steps_taken = 0

        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        
        print(f"Goal Voltage: {self.goal:.4f}")
        self.eng.set_param(f'{self.model_name}/Goal', 'Value', str(self.goal), nargout=0)

        self.eng.set_param(self.model_name, 'FastRestart', 'off', 'LoadInitialState', 'off', nargout=0)
        self.eng.eval(f"out = sim('{self.model_name}', 'StopTime','1e-6', 'SaveFinalState','on', 'StateSaveName','xFinal');"
                      "xFinal = out.xFinal;", nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
        initial_voltage, _ = self.get_data()

        self.prev_error = initial_voltage - self.goal

        if self.enable_plotting:
            for data_list in [self._times, self._voltages, self._goals, self._duties, 
                              self._plus_0_5v, self._minus_0_5v, self._plus_1v, self._minus_1v]:
                data_list.clear()

            self._times.append(0.0)
            self._voltages.append(initial_voltage)
            self._goals.append(self.goal)
            self._duties.append(0.5)
            self._plus_0_5v.append(self.goal + 0.5)
            self._minus_0_5v.append(self.goal - 0.5)
            self._plus_1v.append(self.goal + 1.0)
            self._minus_1v.append(self.goal - 1.0)

            self._update_plot_data()
            if self.voltage_plot_ylim:
                self.ax_voltage.set_ylim(self.voltage_plot_ylim)
            else:
                 self.ax_voltage.autoscale_view()
            self.ax_duty.autoscale_view()
            self.fig.canvas.draw(); self.fig.canvas.flush_events()

        observation = np.array([initial_voltage, self.prev_error, 0.0, self.goal], dtype=np.float32)
        return observation, {}

    def step(self, action):
        """Simulates one agent step."""
        self.steps_taken += 1
        duty_cycle = float(np.clip(action[0], 0.1, 0.9))
        stop_time = self.current_time + (self.dt * self.frame_skip)
        
        self.eng.set_param(f"{self.model_name}/DutyCycleInput", 'Value', str(duty_cycle), nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'LoadInitialState','on', 'InitialState','xFinal',"
            f"'StopTime','{stop_time}', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;", nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)

        voltage, time = self.get_data()
        self.current_time = time
        error = voltage - self.goal
        derivative_error = (error - self.prev_error) / (self.dt * self.frame_skip)
        reward = 1.0 / (1.0 + error**2)
        terminated = bool(self.current_time >= self.max_episode_time)
        truncated = False

        if self.steps_taken > self.grace_period_steps and ((voltage < -5.0) or (voltage > 53.0)):
            reward -= 25.0
            truncated = True

        self.prev_error = error
        observation = np.array([voltage, error, derivative_error, self.goal], dtype=np.float32)

        if self.enable_plotting:
            self._times.append(self.current_time)
            self._voltages.append(voltage)
            self._goals.append(self.goal)
            self._duties.append(duty_cycle)
            self._plus_0_5v.append(self.goal + 0.5)
            self._minus_0_5v.append(self.goal - 0.5)
            self._plus_1v.append(self.goal + 1.0)
            self._minus_1v.append(self.goal - 1.0)
            self._update_plot_data()

        return observation, reward, terminated, truncated, {}

    def _update_plot_data(self):
        """Helper function to update all lines on the plot."""
        self.line_voltage.set_data(self._times, self._voltages)
        self.line_goal.set_data(self._times, self._goals)
        self.line_duty.set_data(self._times, self._duties)
        self.line_plus_0_5v.set_data(self._times, self._plus_0_5v)
        self.line_minus_0_5v.set_data(self._times, self._minus_0_5v)
        self.line_plus_1v.set_data(self._times, self._plus_1v)
        self.line_minus_1v.set_data(self._times, self._minus_1v)
        
        # Selectively autoscale to respect manual y-limits
        self.ax_duty.relim()
        self.ax_duty.autoscale_view()
        self.ax_voltage.relim()
        if not self.voltage_plot_ylim:
            self.ax_voltage.autoscale_view()
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self):
        pass

    def close(self):
        if self.enable_plotting:
            plt.ioff()
            plt.show()
        print("\nMATLAB engine shut down.")
        self.eng.quit()
