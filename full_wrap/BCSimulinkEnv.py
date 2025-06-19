import gym
from gym import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

class BCSimulinkEnv(gym.Env):
    """
    Robust environment for controlling a Buck Converter in Simulink.
    This version correctly implements the xFinal state-passing mechanism,
    provides enhanced logging, and correctly initializes the plot.
    """
    def __init__(self, model_name="bcSim", dt=5e-6, max_episode_time=0.03,
                 grace_period_steps=50,
                 frame_skip=10,
                 enable_plotting=False,
                 target_voltage=30.0):
        
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
        self.steps_taken = 0
        self.current_time = 0.0

        # Defining the action and observation spaces
        # Action space is the duty cycle, which is a value between 0.1 and 0.9
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        # Observation space is the voltage, error, derivative of error, and goal
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        self.prev_error = 0
        self._np_random = None

        if self.enable_plotting:
            self._setup_plot()

    def _setup_plot(self):
        """
        Sets up the live plot for the environment that plots the voltage, goal, and duty cycle.
        """
        print("Setting up the live plot...")
        plt.ion()
        self.fig, (self.ax_voltage, self.ax_duty) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.fig.suptitle('BC Simulink Control (48V Source Voltage)')
        self.line_voltage, = self.ax_voltage.plot([], [], 'b-', label="Actual Voltage")
        self.line_goal,    = self.ax_voltage.plot([], [], 'r--', label="Target Voltage")
        self.ax_voltage.set_ylabel("Voltage (V)")
        self.ax_voltage.legend(loc='best')
        self.ax_voltage.grid(True)
        self.line_duty,    = self.ax_duty.plot([], [], 'g-', label="Duty Cycle (Action)")
        self.ax_duty.set_xlabel("Time (s)")
        self.ax_duty.set_ylabel("Duty Cycle")
        self.ax_duty.set_ylim(0, 1)
        self.ax_duty.legend(loc='best')
        self.ax_duty.grid(True)
        self._times, self._voltages, self._goals, self._duties = [], [], [], []

    def get_data(self):
        """
        Gets the data from the simulation.
        Returns the final voltage and time for each step in the simulation.
        """
        voltage_out = self.eng.eval("out.voltage", nargout=1)
        time_out = self.eng.eval("out.tout", nargout=1)
        final_voltage = voltage_out[-1][0] if not isinstance(voltage_out, float) else voltage_out
        final_time = time_out[-1][0] if not isinstance(time_out, float) else time_out
        return final_voltage, final_time

    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        print("\n--- Episode Reset ---")
        self.current_time = 0.0
        self.steps_taken = 0

        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        # Set a new random goal for the buck converter (e.g., 5V to 45V)


        #self.goal = self.np_random.uniform(low=5.0, high=45.0) 


        print(f"New Goal Voltage: {self.goal:.4f}")
        # Set the goal voltage for the buck converter to a detached constant block within the model (for reference)
        self.eng.set_param(f'{self.model_name}/Goal', 'Value', str(self.goal), nargout=0)

        # Run a tiny simulation to get the initial state at t=0
        self.eng.set_param(self.model_name, 'FastRestart', 'off', 'LoadInitialState', 'off', nargout=0)
        self.eng.eval(f"out = sim('{self.model_name}', 'StopTime','1e-6', 'SaveFinalState','on', 'StateSaveName','xFinal');"
                      "xFinal = out.xFinal;", nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
        initial_voltage, _ = self.get_data()

        self.prev_error = initial_voltage - self.goal

        if self.enable_plotting:
            # Clear data from the previous episode
            for data_list in [self._times, self._voltages, self._goals, self._duties]: data_list.clear()
            self._times.append(0.0)
            self._voltages.append(initial_voltage)
            self._goals.append(self.goal)
            self._duties.append(0.5) # Assume a neutral starting duty cycle for the plot

            # Update the plot to show the starting point
            self.line_voltage.set_data(self._times, self._voltages)
            self.line_goal.set_data(self._times, self._goals)
            self.line_duty.set_data(self._times, self._duties)
            for ax in (self.ax_voltage, self.ax_duty): ax.relim(); ax.autoscale_view()
            self.fig.canvas.draw(); self.fig.canvas.flush_events()

        observation = np.array([initial_voltage, self.prev_error, 0.0, self.goal], dtype=np.float32)
        return observation, {}

    def step(self, action):
        """
        Function that simulates an agent taking one step in the environment.
        Inputs:
            action: The action taken by the agent (generated by SB3).
        Outputs:
            observation: The observation of the environment.
            reward: The reward for the agent.
            terminated: Whether the episode has terminated (i.e. completed).
            truncated: Whether the episode was truncated (i.e. terminated by the environment).
            info: Additional information about the environment.
        """
        self.steps_taken += 1

        # Clip the action to be between 0.1 and 0.9
        duty_cycle = float(np.clip(action[0], 0.1, 0.9))

        # Set the duty cycle for the buck converter
        stop_time = self.current_time + (self.dt * self.frame_skip)
        self.eng.set_param(f"{self.model_name}/DutyCycleInput", 'Value', str(duty_cycle), nargout=0)
        
        # Run the simulation for one step
        self.eng.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'LoadInitialState','on', 'InitialState','xFinal',"
            f"'StopTime','{stop_time}', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;", nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)

        # Get the voltage and time from the simulation
        voltage, time = self.get_data()
        self.current_time = time

        # Calculate the error between the voltage and the goal
        error = voltage - self.goal
        # Calculate the derivative of the error
        step_duration = self.dt * self.frame_skip 
        derivative_error = (error - self.prev_error) / step_duration

        # Reward function focused on minimizing error
        reward = 1.0 / (1.0 + error**2)

        # Check if the episode has terminated
        terminated = bool(self.current_time >= self.max_episode_time)
        truncated = False

        # Penalize for hitting safety limits after a grace period (i.e. if the voltage is too low or too high)
        if self.steps_taken > self.grace_period_steps:
            INPUT_VOLTAGE = 48.0 # Source Voltage Defined in the Simulation
            HARD_LOWER_LIMIT = -5.0
            HARD_UPPER_LIMIT = INPUT_VOLTAGE + 5.0
            if (voltage < HARD_LOWER_LIMIT) or (voltage > HARD_UPPER_LIMIT):
                reward -= 25.0
                truncated = True

        # Update the previous error
        self.prev_error = error

        print(f"reward: {reward:.3f} | voltage: {voltage:.3f} | goal: {self.goal:.3f} | duty: {duty_cycle:.3f}")

        observation = np.array([voltage, error, derivative_error, self.goal], dtype=np.float32)

        if self.enable_plotting:
            self._times.append(self.current_time) # Log the correct time
            self._voltages.append(voltage)
            self._goals.append(self.goal)
            self._duties.append(duty_cycle)
            self.line_voltage.set_data(self._times, self._voltages)
            self.line_goal.set_data(self._times, self._goals)
            self.line_duty.set_data(self._times, self._duties)
            for ax in (self.ax_voltage, self.ax_duty):
                ax.relim()
                ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        return observation, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        if self.enable_plotting:
            plt.close(self.fig)
        print("\nMATLAB engine shut down.")
        self.eng.quit()