import gym
from gym import spaces
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

class BCSimulinkEnv(gym.Env):
    """
    Robust environment for controlling a Buck Converter in Simulink.
    This version includes a boolean flag to switch between a fixed or randomized target voltage.
    """
    def __init__(self, model_name="bcSim", dt=5e-6, max_episode_time=0.1,
                 grace_period_steps=50,
                 frame_skip=10,
                 enable_plotting=False,
                 # --- NEW PARAMETERS FOR GOAL CONTROL ---
                 use_randomized_goal: bool = True,
                 fixed_goal_voltage: float = 30.0,
                 target_voltage_min: float = 27.5,
                 target_voltage_max: float = 32.5):
        
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
        
        # Store the goal-setting strategy and parameters
        self.use_randomized_goal = use_randomized_goal
        self.fixed_goal_voltage = fixed_goal_voltage
        self.target_voltage_min = target_voltage_min
        self.target_voltage_max = target_voltage_max
        self.goal = 30.0 # Initial placeholder
        
        self.steps_taken = 0
        self.current_time = 0.0

        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        self.prev_error = 0
        self.seed()

        if self.enable_plotting:
            self._setup_plot()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _setup_plot(self):
        plt.ion()
        self.fig, (self.ax_voltage, self.ax_duty) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        self.fig.suptitle('BC Simulink Control (48V Source Voltage)')
        
        self.line_voltage, = self.ax_voltage.plot([], [], 'b-', label="Actual Voltage", linewidth=2)
        self.line_goal,    = self.ax_voltage.plot([], [], 'r--', label="Target Voltage")
        self.ax_voltage.set_ylabel("Voltage (V)"); self.ax_voltage.legend(loc='best'); self.ax_voltage.grid(True)
        
        self.line_duty,    = self.ax_duty.plot([], [], 'm-', label="Duty Cycle (Action)")
        self.ax_duty.set_xlabel("Time (s)"); self.ax_duty.set_ylabel("Duty Cycle"); self.ax_duty.set_ylim(0, 1)
        self.ax_duty.legend(loc='best'); self.ax_duty.grid(True)
        
        self._times, self._voltages, self._goals, self._duties = [], [], [], []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
            
        self.current_time = 0.0
        self.steps_taken = 0

        # --- KEY CHANGE: Set goal based on the boolean flag ---
        if self.use_randomized_goal:
            self.goal = self.np_random.uniform(self.target_voltage_min, self.target_voltage_max)
        else:
            self.goal = self.fixed_goal_voltage
        
        self.eng.set_param(f'{self.model_name}/Goal', 'Value', str(self.goal), nargout=0)

        self.eng.set_param(self.model_name, 'FastRestart', 'off', 'LoadInitialState', 'off', nargout=0)
        self.eng.eval(f"out = sim('{self.model_name}', 'StopTime','1e-6', 'SaveFinalState','on', 'StateSaveName','xFinal');"
                      "xFinal = out.xFinal;", nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
        initial_voltage, _ = self.get_data()

        self.prev_error = initial_voltage - self.goal

        if self.enable_plotting:
            for data_list in [self._times, self._voltages, self._goals, self._duties]:
                data_list.clear()
            self._times.append(0.0); self._voltages.append(initial_voltage)
            self._goals.append(self.goal); self._duties.append(0.5)
            self._update_plot_data()

        observation = np.array([initial_voltage, self.prev_error, 0.0, self.goal], dtype=np.float32)
        return observation, {}

    def step(self, action):
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
        step_duration = self.dt * self.frame_skip
        derivative_error = (error - self.prev_error) / step_duration
        reward = 1.0 / (1.0 + error**2)
        terminated = bool(self.current_time >= self.max_episode_time)
        truncated = False

        if self.steps_taken > self.grace_period_steps:
            if (voltage < -5.0) or (voltage > 53.0):
                reward -= 25.0
                truncated = True

        self.prev_error = error
        observation = np.array([voltage, error, derivative_error, self.goal], dtype=np.float32)

        if self.enable_plotting:
            self._times.append(self.current_time); self._voltages.append(voltage)
            self._goals.append(self.goal); self._duties.append(duty_cycle)
            self._update_plot_data()

        return observation, reward, terminated, truncated, {}
    
    def get_data(self):
        voltage_out = self.eng.eval("out.voltage", nargout=1)
        time_out = self.eng.eval("out.tout", nargout=1)
        final_voltage = voltage_out[-1][0] if not isinstance(voltage_out, float) else voltage_out
        final_time = time_out[-1][0] if not isinstance(time_out, float) else time_out
        return final_voltage, final_time

    def _update_plot_data(self):
        self.line_voltage.set_data(self._times, self._voltages)
        self.line_goal.set_data(self._times, self._goals)
        self.line_duty.set_data(self._times, self._duties)
        for ax in (self.ax_voltage, self.ax_duty):
            ax.relim(); ax.autoscale_view()
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def render(self): pass
    def close(self):
        if self.enable_plotting: plt.ioff(); plt.show()
        print("\nMATLAB engine shut down."); self.eng.quit()