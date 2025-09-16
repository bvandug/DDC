import gymnasium as gym
from gymnasium import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt


class DiscretizeActionWrapper(gym.ActionWrapper):
    """A wrapper to discretize a continuous action space for DQN.

    This class takes a Gymnasium environment with a continuous action space
    and converts it into a discrete action space with a specified number of
    bins. This is necessary for using algorithms like DQN that require a

    discrete action set.

    Attributes:
        n_bins (int): The number of discrete actions.
        continuous_actions (np.ndarray): The array of continuous action values
            that correspond to each discrete action.
    """
    def __init__(self, env, n_bins=17):
        """Initializes the DiscretizeActionWrapper.

        Args:
            env (gym.Env): The Gymnasium environment to wrap.
            n_bins (int): The number of discrete bins to create.
        """
        super().__init__(env)
        self.n_bins = n_bins
        self.action_space = spaces.Discrete(self.n_bins)
        # Create a set of evenly spaced continuous actions from the original space
        self.continuous_actions = np.linspace(
            self.env.action_space.low[0],
            self.env.action_space.high[0],
            self.n_bins
        )

    def action(self, action):
        """Translates a discrete action into its continuous equivalent.

        This method is called by the wrapper to convert the agent's discrete
        action into a continuous value that can be passed to the underlying
        environment.

        Args:
            action (int): The discrete action chosen by the agent.

        Returns:
            np.ndarray: The corresponding continuous action as a numpy array.
        """
        continuous_action = self.continuous_actions[action]
        return np.array([continuous_action], dtype=np.float32)


class BCSimulinkEnv(gym.Env):
    """A Gymnasium environment for controlling a Buck Converter in Simulink.

    This environment interfaces with a MATLAB Simulink model to simulate a
    DC-DC buck converter. It allows a reinforcement learning agent to control
    the converter's duty cycle to regulate the output voltage. The environment
    supports configurable sensor noise to simulate real-world conditions.

    Attributes:
        eng (matlab.engine.MatlabEngine): The active MATLAB engine instance.
        model_name (str): The name of the Simulink model file.
        dt (float): The fundamental time step of the physics simulation.
        frame_skip (int): The number of `dt` steps per agent action.
        goal (float): The current target voltage for the episode.
    """
    def __init__(self, model_name="bcSim", dt=5e-6, max_episode_time=0.1,
                 grace_period_steps=50,
                 frame_skip=10,
                 enable_plotting=False,
                 use_randomized_goal=False,
                 fixed_goal_voltage=30.0,
                 target_voltage_min=28.5,
                 target_voltage_max=31.5,
                 voltage_noise_std: float = 0.0):
        """Initializes the BCSimulinkEnv.

        Args:
            model_name (str): The name of the Simulink model file (without .slx).
            dt (float): The fundamental time step for the physics simulation (s).
            max_episode_time (float): The total duration of one episode in seconds.
            grace_period_steps (int): Initial steps to ignore termination.
            frame_skip (int): The number of physics simulations per agent step.
            enable_plotting (bool): If True, a live plot of the state is displayed.
            use_randomized_goal (bool): If True, randomize target voltage.
            fixed_goal_voltage (float): Target voltage if randomization is off.
            target_voltage_min (float): Min value for randomized target voltage.
            target_voltage_max (float): Max value for randomized target voltage.
            voltage_noise_std (float): Std deviation of Gaussian sensor noise.
        """
        super().__init__()

        print("Starting MATLAB engine...")
        # Start a local, non-graphical MATLAB session
        self.eng = matlab.engine.start_matlab("-nodesktop -nosplash")
        print(f"Loading {model_name}...")
        self.eng.load_system(model_name, nargout=0)
        # Enable Fast Restart for faster simulation iterations
        self.eng.set_param(model_name, 'FastRestart', 'on', nargout=0)

        # Initialize environment parameters
        self.model_name = model_name
        self.dt = dt
        self.max_episode_time = max_episode_time + (dt * 0.5)
        self.frame_skip = frame_skip
        self.enable_plotting = enable_plotting
        self.grace_period_steps = grace_period_steps
        self.voltage_noise_std = voltage_noise_std

        # Store goal-setting strategy
        self.use_randomized_goal = use_randomized_goal
        self.fixed_goal_voltage = fixed_goal_voltage
        self.target_voltage_min = target_voltage_min
        self.target_voltage_max = target_voltage_max
        self.goal = 30.0

        # Internal state variables
        self.steps_taken = 0
        self.current_time = 0.0
        self.prev_error = 0
        self.np_random, _ = gym.utils.seeding.np_random()

        # Define action and observation spaces
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,),
                                       dtype=np.float32)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,),
                                            dtype=np.float32)

        if self.enable_plotting:
            self._setup_plot()

    def _setup_plot(self):
        """Sets up the live plot for rendering."""
        print("Setting up the live plot...")
        plt.ion()
        self.fig, (self.ax_voltage, self.ax_duty) = plt.subplots(
            2, 1, figsize=(12, 9), sharex=True
        )
        self.fig.suptitle('BC Simulink Control (48V Source Voltage)')

        # Voltage Plot setup
        self.line_voltage, = self.ax_voltage.plot([], [], 'b-',
                                                   label="Actual Voltage",
                                                   linewidth=2)
        self.line_goal, = self.ax_voltage.plot([], [], 'r--',
                                               label="Target Voltage")
        self.line_plus_0_5v, = self.ax_voltage.plot([], [], 'g:',
                                                     label="±0.5V Tolerance")
        self.line_minus_0_5v, = self.ax_voltage.plot([], [], 'g:')
        self.line_plus_1v, = self.ax_voltage.plot([], [], 'k:',
                                                   label="±1.0V Tolerance")
        self.line_minus_1v, = self.ax_voltage.plot([], [], 'k:')

        self.ax_voltage.set_ylabel("Voltage (V)")
        self.ax_voltage.legend(loc='best')
        self.ax_voltage.grid(True)

        # Duty Cycle Plot setup
        self.line_duty, = self.ax_duty.plot([], [], 'm-',
                                            label="Duty Cycle (Action)")
        self.ax_duty.set_xlabel("Time (s)")
        self.ax_duty.set_ylabel("Duty Cycle")
        self.ax_duty.set_ylim(0, 1)
        self.ax_duty.legend(loc='best')
        self.ax_duty.grid(True)

        # Initialize data lists for plotting
        self._times, self._voltages, self._goals, self._duties = [], [], [], []
        self._plus_0_5v, self._minus_0_5v = [], []
        self._plus_1v, self._minus_1v = [], []

    def set_goal_voltage(self, voltage):
        """Allows external scripts to set the target voltage for evaluation."""
        self.goal = voltage

    def get_data(self):
        """Retrieves the final voltage and time from the last Simulink run."""
        voltage_out = self.eng.eval("out.voltage", nargout=1)
        time_out = self.eng.eval("out.tout", nargout=1)
        final_voltage = voltage_out[-1][0] if not isinstance(voltage_out,
                                                             float) else voltage_out
        final_time = time_out[-1][0] if not isinstance(time_out,
                                                       float) else time_out
        return final_voltage, final_time

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode.

        Returns:
            tuple: A tuple containing the initial observation and an info dict.
        """
        super().reset(seed=seed)

        self.current_time = 0.0
        self.steps_taken = 0

        # Set the goal for the new episode
        if self.use_randomized_goal:
            self.goal = self.np_random.uniform(self.target_voltage_min,
                                               self.target_voltage_max)
        else:
            self.goal = self.fixed_goal_voltage

        # Update the goal value in the Simulink model
        self.eng.set_param(f'{self.model_name}/Goal', 'Value', str(self.goal),
                           nargout=0)

        # Run a minimal simulation to get a consistent initial state
        self.eng.set_param(self.model_name, 'FastRestart', 'off',
                           'LoadInitialState', 'off', nargout=0)
        self.eng.eval(f"out = sim('{self.model_name}', 'StopTime','1e-6', "
                      f"'SaveFinalState','on', 'StateSaveName','xFinal');"
                      f" xFinal = out.xFinal;", nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
        initial_voltage, _ = self.get_data()

        # Add noise to the initial reading to create the agent's observation
        noisy_initial_voltage = initial_voltage + self.np_random.normal(
            0, self.voltage_noise_std
        )
        self.prev_error = noisy_initial_voltage - self.goal
        observation = np.array([noisy_initial_voltage, self.prev_error, 0.0,
                                self.goal], dtype=np.float32)

        # Reset plot data if rendering is enabled
        if self.enable_plotting:
            for data_list in [self._times, self._voltages, self._goals,
                              self._duties, self._plus_0_5v, self._minus_0_5v,
                              self._plus_1v, self._minus_1v]:
                data_list.clear()
            self._times.append(0.0)
            self._voltages.append(initial_voltage)
            self._goals.append(self.goal)
            self._duties.append(0.5)
            self._update_plot_tolerances()
            self._update_plot_data()

        info = {}
        return observation, info

    def step(self, action):
        """Advances the simulation by one agent step.

        Args:
            action (np.ndarray): The action from the agent (duty cycle).

        Returns:
            tuple: A tuple containing the new observation, reward, terminated
            flag, truncated flag, and an info dictionary.
        """
        self.steps_taken += 1
        duty_cycle = float(np.clip(action[0], 0.1, 0.9))
        stop_time = self.current_time + (self.dt * self.frame_skip)

        # Update Simulink parameters and run the simulation for one step
        self.eng.set_param(f"{self.model_name}/DutyCycleInput", 'Value',
                           str(duty_cycle), nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
        self.eng.eval(f"out = sim('{self.model_name}', 'LoadInitialState','on',"
                      f" 'InitialState','xFinal', 'StopTime','{stop_time}',"
                      f" 'SaveFinalState','on', 'StateSaveName','xFinal');"
                      f" xFinal = out.xFinal;", nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)

        # Get the true, clean voltage from the simulation
        true_voltage, time = self.get_data()
        self.current_time = time

        # 1. Calculate reward and termination based on the TRUE physical state
        true_error = true_voltage - self.goal
        reward = 1.0 / (1.0 + true_error ** 2) - 0.01

        terminated = False
        if self.steps_taken > self.grace_period_steps and \
                not (-5.0 < true_voltage < 53.0):
            reward -= 25.0
            terminated = True

        # 2. Create the NOISY observation for the agent
        noisy_voltage = true_voltage + self.np_random.normal(
            0, self.voltage_noise_std
        )
        noisy_error = noisy_voltage - self.goal
        step_duration = self.dt * self.frame_skip
        derivative_error = (noisy_error - self.prev_error) / step_duration

        # Update prev_error with the noisy one for the next derivative
        self.prev_error = noisy_error
        observation = np.array([noisy_voltage, noisy_error, derivative_error,
                                self.goal], dtype=np.float32)

        # Check for truncation (end of episode time)
        truncated = bool(self.current_time >= self.max_episode_time)

        # Update plot data if rendering is enabled
        if self.enable_plotting:
            self._times.append(self.current_time)
            self._voltages.append(true_voltage)  # Plot the true voltage
            self._goals.append(self.goal)
            self._duties.append(duty_cycle)
            self._update_plot_tolerances()
            self._update_plot_data()

        info = {'true_voltage': true_voltage}
        return observation, reward, terminated, truncated, info

    def _update_plot_tolerances(self):
        """Helper to update tolerance band data for plotting."""
        self._plus_0_5v.append(self.goal + 0.5)
        self._minus_0_5v.append(self.goal - 0.5)
        self._plus_1v.append(self.goal + 1.0)
        self._minus_1v.append(self.goal - 1.0)

    def _update_plot_data(self):
        """Helper function to update all lines on the plot."""
        self.line_voltage.set_data(self._times, self._voltages)
        self.line_goal.set_data(self._times, self._goals)
        self.line_duty.set_data(self._times, self._duties)
        self.line_plus_0_5v.set_data(self._times, self._plus_0_5v)
        self.line_minus_0_5v.set_data(self._times, self._minus_0_5v)
        self.line_plus_1v.set_data(self._times, self._plus_1v)
        self.line_minus_1v.set_data(self._times, self._minus_1v)

        for ax in (self.ax_voltage, self.ax_duty):
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render(self):
        """Live rendering is handled in the step method, so this is a no-op."""
        pass

    def close(self):
        """Shuts down the MATLAB engine and closes the plot."""
        if self.enable_plotting:
            plt.ioff()
            plt.show()
        print("\nMATLAB engine shut down.")
        self.eng.quit()
