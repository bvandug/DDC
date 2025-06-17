import gym
from gym import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

class BBCSimulinkEnv(gym.Env):
    """
    Final efficient and stable version of the BBC Simulink environment.
    There are 3 main phases of the training of an agent:
    1. Random exploration (grace_period_steps): the agent randomly explores the environment without learning in order to create a vast amount of data
    2. Survival: the agent learns to survive in the environment by learning to avoid the hard limits
    3. Learning (learning_starts): the agent learns to reach the goal voltage by learning to control the duty cycle
    """
    def __init__(self, model_name="bbcSim", dt=5e-6, max_episode_time=0.03,
                 grace_period_steps=50,
                 frame_skip=10,
                 enable_plotting=False):
        """
        Args:
            model_name (str): The name of the Simulink model file.
            dt (float): The base simulation time step.
            max_episode_time (float): The maximum simulation time for one episode.
            grace_period_steps (int): A SHORT number of initial steps to ignore penalties.
            frame_skip (int): The number of simulation steps to run for each single agent action.
            enable_plotting (bool): If True, renders the live plot. Should be False for training.
        """
        # --- Initialization ---
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()
        print(f"Loading {model_name}...")
        self.eng.load_system(model_name, nargout=0)
        self.eng.set_param(model_name, 'FastRestart', 'on', nargout=0)

        self.model_name = model_name
        self.dt = dt
        self.max_episode_time = max_episode_time
        self.frame_skip = frame_skip
        self.enable_plotting = enable_plotting
        self.grace_period_steps = grace_period_steps
        self.steps_taken = 0
        self.current_time = 0.0

        # Define Gym action and observation spaces
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        self.prev_error = 0
        self._np_random = None

        # --- Conditional Plot Setup ---
        if self.enable_plotting:
            print("Setting up the live plot...")
            plt.ion()
            self.fig, (self.ax_voltage, self.ax_duty) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            self.fig.suptitle('BBC Simulink Control')
            self.line_voltage, = self.ax_voltage.plot([], [], 'b-', label="Actual Voltage")
            self.line_goal,    = self.ax_voltage.plot([], [], 'r--', label="Target Voltage")
            self.ax_voltage.set_ylabel("Voltage (V)"); self.ax_voltage.legend(loc='best'); self.ax_voltage.grid(True)
            self.line_duty,    = self.ax_duty.plot([], [], 'g-', label="Duty Cycle (Action)")
            self.ax_duty.set_xlabel("Time (s)"); self.ax_duty.set_ylabel("Duty Cycle"); self.ax_duty.set_ylim(0, 1); self.ax_duty.legend(loc='best'); self.ax_duty.grid(True)

        self._times, self._voltages, self._goals, self._duties = [], [], [], []

    def get_data(self):
        voltage_out = self.eng.eval("out.voltage", nargout=1)
        time_out = self.eng.eval("out.tout", nargout=1)
        final_voltage = voltage_out[-1][0] if not isinstance(voltage_out, float) else voltage_out
        final_time = time_out[-1][0] if not isinstance(time_out, float) else time_out
        return final_voltage, final_time

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n--- Episode Reset ---")
        self.current_time = 0.0
        self.steps_taken = 0  # Reset counter for every new episode

        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        self.goal = self.np_random.uniform(low=-49.0, high=-28.0)
        print(f"New Goal Voltage: {self.goal:.4f}")
        self.eng.set_param(f'{self.model_name}/Goal', 'Value', str(self.goal), nargout=0)

        self.eng.set_param(self.model_name, 'FastRestart', 'off', 'LoadInitialState', 'off', nargout=0)
        self.eng.eval(f"out = sim('{self.model_name}', 'StopTime','1e-6', 'SaveFinalState','on', 'StateSaveName','xFinal');" "xFinal = out.xFinal;", nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)

        initial_voltage, _ = self.get_data()
        self.prev_error = initial_voltage - self.goal

        if self.enable_plotting:
            for data_list in [self._times, self._voltages, self._goals, self._duties]: data_list.clear()
            for line in (self.line_voltage, self.line_goal, self.line_duty): line.set_data([], [])
            for ax in (self.ax_voltage, self.ax_duty): ax.relim(); ax.autoscale_view()
            self.fig.canvas.draw(); self.fig.canvas.flush_events()

        observation = np.array([initial_voltage, self.prev_error, 0.0, self.goal], dtype=np.float32)
        return observation, {}

    def step(self, action):
        self.steps_taken += 1
        duty_cycle = float(np.clip(action[0], 0.1, 0.9))

        stop_time = self.current_time + (self.dt * self.frame_skip)
        self.eng.set_param(f"{self.model_name}/DutyCycleInput", 'Value', str(duty_cycle), nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
        self.eng.eval(f"out = sim('{self.model_name}', 'LoadInitialState','on', 'InitialState','xFinal'," f"'StopTime','{stop_time}', 'SaveFinalState','on', 'StateSaveName','xFinal');" "xFinal = out.xFinal;", nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)

        voltage, t = self.get_data()
        self.current_time = t

        error = voltage - self.goal
        derivative_error = error - self.prev_error

        reward_accuracy = -0.01 * (error**2)
        progress = abs(self.prev_error) - abs(error)
        reward_progress = 5 * progress
        reward_efficiency = -1 * (duty_cycle**2)

        reward = reward_accuracy + reward_progress + reward_efficiency

        terminated = bool(t >= self.max_episode_time)
        truncated = False

        # Grace period check for safety limits
        if self.steps_taken > self.grace_period_steps: # Goal Voltage is between -49 and -28
            HARD_LOWER_LIMIT = -90.0 # 41 below the goal voltage range
            HARD_UPPER_LIMIT = -15.0 # 13 above the goal voltage range
            SOFT_LOWER_LIMIT = -52.0 # 3 below the goal voltage range
            SOFT_UPPER_LIMIT = -25.0 # 3 above the goal voltage range

            if (voltage < HARD_LOWER_LIMIT):
                penalty = 20 + 10 * abs(voltage - HARD_LOWER_LIMIT)  # Smoothly increase penalty as violation worsens
                reward -= penalty
                truncated = True
            elif (voltage > HARD_UPPER_LIMIT):
                penalty = 20 + 10 * abs(voltage - HARD_UPPER_LIMIT)
                reward -= penalty
                truncated = True
            if (SOFT_LOWER_LIMIT < voltage < HARD_LOWER_LIMIT):
                reward -= 5 * (HARD_LOWER_LIMIT - voltage)  # Gradually penalize based on distance
            elif (SOFT_UPPER_LIMIT > voltage > HARD_UPPER_LIMIT):
                reward -= 5 * (voltage - HARD_UPPER_LIMIT)

        reward = np.tanh(reward / 25.0) * 25.0
        self.prev_error = error
        print(f"reward: {reward:.3f} | acc: {reward_accuracy:.3f} | prog: {reward_progress:.3f} | eff: {reward_efficiency:.3f} | voltage: {voltage:.3f}")

        observation = np.array([voltage, error, derivative_error, self.goal], dtype=np.float32)

        if self.enable_plotting:
            self._times.append(t); self._voltages.append(voltage); self._goals.append(self.goal); self._duties.append(duty_cycle)
            self.line_voltage.set_data(self._times, self._voltages); self.line_goal.set_data(self._times, self._goals); self.line_duty.set_data(self._times, self._duties)
            for ax in (self.ax_voltage, self.ax_duty): ax.relim(); ax.autoscale_view()
            self.fig.canvas.draw(); self.fig.canvas.flush_events()

        return observation, reward, terminated, truncated, {}

    def close(self):
        if self.enable_plotting:
            plt.close(self.fig)
        print("\nMATLAB engine shut down.")
        self.eng.quit()