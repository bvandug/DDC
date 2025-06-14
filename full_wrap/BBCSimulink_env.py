import gym
from gym import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

class BBCSimulinkEnv(gym.Env):
    """
    A universal Gym environment for the Buck-Boost Converter Simulink model,
    compatible with multiple stable-baselines3 agents (PPO, SAC, TD3, etc.).
    
    This corrected version includes the 'goal' in the observation space to allow
    the agent to generalize across different target voltages.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, model_name="bbcSim", dt=1e-5, max_episode_time=0.015):
        """
        dt: Time step for the simulation
        - original: 5e-6
        - try: 1e-5
        If doesnt work:
        - try max episode time: 0.012
        - try max episode time: 0.01
        max_episode_time: Maximum time for the episode
        Steps per episode = max_episode_time / dt (e.g. 0.015 / 5e-6 = 3000 steps per episode)
        Total episodes = Total timesteps / steps per episode (e.g. 25000 / 3000 = 8.33 episodes)
        """
        # --- Initialization ---
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()
        print(f"Loading {model_name}...")
        self.eng.load_system(model_name, nargout=0)
        print("Model loaded!")
        # CRITICAL: Ensure Fast Restart is enabled in the Simulink model itself.
        self.eng.set_param(model_name, 'FastRestart', 'on', nargout=0)

        self.model_name = model_name
        self.dt = dt
        self.max_episode_time = max_episode_time
        self.current_time = 0.0

        # Define the action space (duty cycle)
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)

        # Define the observation space with 4 elements: [voltage, error, derivative_error, goal]
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        self.prev_error = 0
        self._np_random = None # For seeding

        # Setup the live plot
        print("Setting up the live plot...")
        plt.ion()
        self.fig, (self.ax_voltage, self.ax_duty) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.fig.suptitle('BBC Simulink Control')

        self.line_voltage, = self.ax_voltage.plot([], [], 'b-', label="Actual Voltage")
        self.line_goal,    = self.ax_voltage.plot([], [], 'r--', label="Target Voltage")
        self.line_duty,    = self.ax_duty.plot([], [], 'g-', label="Duty Cycle (Action)")

        self.ax_voltage.set_ylabel("Voltage (V)")
        self.ax_voltage.legend(loc='best')
        self.ax_voltage.grid(True)

        self.ax_duty.set_xlabel("Time (s)")
        self.ax_duty.set_ylabel("Duty Cycle")
        self.ax_duty.set_ylim(0, 1)
        self.ax_duty.legend(loc='best')
        self.ax_duty.grid(True)

        self._times, self._voltages, self._goals, self._duties = [], [], [], []

    def get_data(self):
        """
        Collects and returns the workspace data.
        """
        voltage_out = self.eng.eval("out.voltage", nargout=1)
        time_out = self.eng.eval("out.tout", nargout=1)

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
        Resets the environment.
        """
        super().reset(seed=seed)

        print("\n--- Episode Reset ---")
        self.current_time = 0.0
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)

        # Set a random goal that spans both conditional ranges
        self.goal = self.np_random.uniform(low=-49.0, high=-28.0)

        print(f"New Goal Voltage: {self.goal:.4f}")

        self.eng.set_param(f'{self.model_name}/Goal', 'Value', str(self.goal), nargout=0)

        self.eng.set_param(self.model_name, 'FastRestart', 'off', 'LoadInitialState', 'off', nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'StopTime','1e-6', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0
        )
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)

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

        # Include the goal in the observation array
        observation = np.array([initial_voltage, self.prev_error, 0.0, self.goal], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and done flag.
        """
        proposed_duty_cycle = action[0]

        # --- UPDATE 1: REMOVE FLAWED CONDITIONAL LOGIC ---
        # The old if/else logic was incorrect. A single, wide range allows the
        # agent to learn the true optimal duty cycle for any goal.
        min_duty = 0.1
        max_duty = 0.9
        duty_cycle = float(np.clip(proposed_duty_cycle, min_duty, max_duty))

        # Set the parameter and run the simulation for one time step
        self.eng.set_param(f"{self.model_name}/DutyCycleInput", 'Value', str(duty_cycle), nargout=0)
        stop_time = self.current_time + self.dt
        
        self.eng.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'LoadInitialState','on', 'InitialState','xFinal',"
            f"'StopTime','{stop_time}', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0
        )
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)

        # Get results and calculate reward
        voltage, t = self.get_data()
        self.current_time = t

        error = voltage - self.goal
        derivative_error = error - self.prev_error
        self.prev_error = error

        # Include the goal in the observation array
        observation = np.array([voltage, error, derivative_error, self.goal], dtype=np.float32)

        # REWARD FUNCTION COMPONENTS
        # 1. ACCURACY (negative reward)
        # --- UPDATE: Use a logarithmic penalty instead of squared ---
        # This is much less sensitive to huge initial errors and encourages exploration.
        # The +1 prevents log(0) which is undefined.
        reward_error = -np.log(abs(error) + 1)

        # 2. EFFICIENCY (negative reward)
        reward_duty = -0.5 * (duty_cycle ** 2)

        # 3. RESPONSIVENESS (positive reward)
        progress = abs(self.prev_error) - abs(error)
        reward_progress = progress * 10

        # 4. STABILITY (negative reward)
        # if approaches too slow or sluggish, change to -0.5
        reward_instability = -1.0 * (derivative_error ** 2)

        # Combine reward components
        reward = reward_error + reward_duty + reward_progress + reward_instability

        # The print statement below is for debugging and can be commented out for faster training.
        print(f"Step {len(self._times):<4} | Action (Duty Cycle): {duty_cycle:.4f} -> Voltage: {voltage:.4f} | Goal: {self.goal:.2f} | Reward: {reward:.4f}")

        terminated = bool(t >= self.max_episode_time)
        truncated = False
        info = {}

        # Update plot data
        self._times.append(t); self._voltages.append(voltage); self._goals.append(self.goal); self._duties.append(duty_cycle)
        self.line_voltage.set_data(self._times, self._voltages)
        self.line_goal.set_data(self._times, self._goals)
        self.line_duty.set_data(self._times, self._duties)

        for ax in (self.ax_voltage, self.ax_duty):
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        #print(f"Step {len(self._times):<4} | Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        plt.close(self.fig)
        print("\nMATLAB engine shut down.")
        self.eng.quit()