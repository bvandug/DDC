import gym
from gym import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

class BBCSimulinkEnv(gym.Env):
    """
    A Gym environment for the Buck-Boost Converter Simulink model,
    updated for compatibility with modern stable-baselines3 and Gymnasium API.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, model_name="bbcSim", dt=5e-6, max_episode_time=0.15):
        # --- Initialization ---
        print("Starting MATLAB engine...")
        # Start the MATLAB engine and load the model
        self.eng = matlab.engine.start_matlab()
        print(f"Loading {model_name}...")
        self.eng.load_system(model_name, nargout=0)
        print("Model loaded!")
        self.eng.set_param(model_name, 'FastRestart', 'on', nargout=0)

        self.model_name = model_name
        self.dt = dt
        self.max_episode_time = max_episode_time
        self.current_time = 0.0

        # Define action and observation spaces
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32) # Defining the action space for the duty cycle (The actions that can be taken), shape = (1,) means a single value
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(3,), dtype=np.float32) # Defining the observation space for the voltage, error, and derivative of the error, shape = (3,) means three values
        
        self.prev_error = 0
        self._np_random = None # For seeding

        #Setup the live plot
        print("Setting up the live plot...")
        plt.ion() # Turn on interactive mode for plotting
        self.fig, (self.ax_voltage, self.ax_duty) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.fig.suptitle('BBC Simulink Control')
        
        # Create empty Line2D objects for each signal
        self.line_voltage, = self.ax_voltage.plot([], [], 'b-', label="Actual Voltage")
        self.line_goal,    = self.ax_voltage.plot([], [], 'r--', label="Target Voltage")
        self.line_duty,    = self.ax_duty.plot([], [], 'g-', label="Duty Cycle (Action)")

        # Configure voltage plot
        self.ax_voltage.set_ylabel("Voltage (V)")
        self.ax_voltage.legend(loc='best')
        self.ax_voltage.grid(True)
        
        # Configure duty cycle plot
        self.ax_duty.set_xlabel("Time (s)")
        self.ax_duty.set_ylabel("Duty Cycle")
        self.ax_duty.set_ylim(0, 1) # Duty cycle is between 0 and 1
        self.ax_duty.legend(loc='best')
        self.ax_duty.grid(True)
        
        # Buffers to store data for plotting
        self._times = []
        self._voltages = []
        self._goals = []
        self._duties = []

    def get_data(self):
        """
        Collects and returns the workspace data, handling both
        single-value and multi-value returns from MATLAB.
        Returns:
            voltage_lst (List[float]) : Observed voltages.
            time_lst  (List[float]) : Simulation times.
        """

        voltage_out = self.eng.eval("out.voltage", nargout=1)
        time_out = self.eng.eval("out.tout", nargout=1)

        # if it's a bare float, wrap it so our loop works
        if isinstance(voltage_out, float):
            final_voltage = voltage_out
        else:
            final_voltage = voltage_out[-1][0] 

        if isinstance(time_out, float):
            final_time = time_out
        else:
            final_time = time_out[-1][0]

        return final_voltage, final_time
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        Returns:
            observation (np.ndarray): The initial observation.
            info (dict): Additional information.
        """
        super().reset(seed=seed) # Important for seeding in wrappers
        
        print("\n--- Episode Reset ---")
        # Reset simulation
        self.current_time = 0.0
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)

        # Set random goal (this will now be deterministic if seeded)
        self.goal = self.np_random.uniform(low=-32.0, high=-28.0)
        
        print(f"New Goal Voltage: {self.goal:.4f}")
        
        # Set the Goal constant within the model, it is not used within the model but is rather used so python can fetch the goal voltage
        self.eng.set_param(f'{self.model_name}/Goal', 'Value', str(self.goal), nargout=0)
        
        # Initialize simulation state ('xFinal')
        self.eng.set_param(self.model_name, 'FastRestart', 'off', 'LoadInitialState', 'off', nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'StopTime','1e-6', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0
        )
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)

        # Get the initial voltage and error
        initial_voltage, _ = self.get_data()
        self.prev_error = initial_voltage - self.goal

        # Clear plot for new episode
        for data_list in [self._times, self._voltages, self._goals, self._duties]:
            data_list.clear()
        for line in (self.line_voltage, self.line_goal, self.line_duty):
            line.set_data([], [])

        for ax in (self.ax_voltage, self.ax_duty):
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # The environment must return the observation and an info dictionary
        observation = np.array([initial_voltage, self.prev_error, 0.0], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and done flag.
        Args:
            action (np.ndarray): The action taken by the agent.
        Returns:
            observation (np.ndarray): The next state.
            reward (float): The reward for the action.
            terminated (bool): Whether the episode has terminated.
        """
        # Clip the action to the action space (get the duty cycle from SB3)
        duty_cycle = float(np.clip(action[0], self.action_space.low, self.action_space.high))
        
        print(f"Step {len(self._times):<4} | Action (Duty Cycle): {duty_cycle:.4f}", end="")

        # Set the parameter of the duty cycle constant input block to the duty cycle
        self.eng.set_param(f"{self.model_name}/DutyCycleInput", 'Value', str(duty_cycle), nargout=0)

        # Run simulation for one time step
        stop_time = self.current_time + self.dt
        self.eng.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'LoadInitialState','on', 'InitialState','xFinal',"
            f"'StopTime','{stop_time}', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0
        )
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
        
        # Get results
        voltage, t = self.get_data()
        self.current_time = t
        
        # Calculate observation, reward, and done
        error = voltage - self.goal
        derivative_error = (error - self.prev_error) / self.dt if self.dt > 0 else 0
        self.prev_error = error
        observation = np.array([voltage, error, derivative_error], dtype=np.float32)
        
        # Use reward shaping for the reward function
        # provides the agent with continuous feedback at every step, rather than 0 if far away, so they need to get closer to the goal and stay closer to the goal
        reward = -(error ** 2) - 0.01 * (duty_cycle ** 2) # it is the error penalization - a penalty for the control effort

        #error_margin = 10 
        #if (self.goal - error_margin) <= voltage <= (self.goal + error_margin):
        #    reward = (error_margin - abs(error))**2 / error_margin
        #else:
        #    reward = 0
        # Some reward functions to try:
        # reward = -(error ** 2) -> Penalizes large errors more heavily, encourages precision
        # reward = reward = -(error ** 2) - 0.01 * (action ** 2) -> balances voltage tracking and control smoothness

        print(f" -> Voltage: {voltage:.4f} | Reward: {reward:.4f}")

        # Check if the episode has terminated or been truncated
        terminated = bool(t >= self.max_episode_time)
        truncated = False # Not using truncation based on time limits here
        info = {}
        
        # Update the plot
        self._times.append(t)
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
        
        # Step must now return 5 values for gymnasium compatibility
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        # The step method already handles rendering the plot. But overwriting needed for gymnasium compatibility
        pass

    def close(self):
        plt.close(self.fig) # Close the plot figure
        self.eng.quit()