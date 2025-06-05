# simulink_env.py

import gym
from gym import spaces
import numpy as np
import matlab.engine
import os # Keep for potential future use, though not for per-step state files

class SimulinkEnv(gym.Env):
    """
    A Gym wrapper around your Pendulum-on-Cart Simulink model,
    using FastRestart and workspace variables for state persistence.
    """
    metadata = {'render.modes': []}
    # Use a consistent MATLAB variable name for the operating point
    OPERATING_POINT_VAR_NAME = 'envOpVar'

    def __init__(self,
                 model_name: str = "PendCart",
                 agent_block: str = "PendCart/DRL", # Assuming this is where action is applied
                 dt: float = 0.02,
                 max_episode_time: float = 5,
                 angle_threshold: float = np.pi/2):
        super().__init__()
        global eng
        # Ensure MATLAB engine is started (idempotent start)
        if 'eng' not in globals() or eng is None:
            print("Starting MATLAB engine...")
            eng = matlab.engine.start_matlab()
            # Add the current directory to MATLAB's path if model is local
            # eng.addpath(os.getcwd(), nargout=0) # Potentially useful
        else:
            print("Using existing MATLAB engine.")

        self.model_name       = model_name
        self.agent_block_path = agent_block # Full path to the agent's action block
        self.dt               = dt
        self.current_time     = 0.0
        self.max_episode_time = max_episode_time
        self.angle_threshold  = angle_threshold
        
        # This will store the last operating point from MATLAB, to be passed to InitialState
        self.last_operating_point = None 

        # --- Configure Model for FastRestart and State Saving (One-time setup) ---
        print(f"Configuring model '{self.model_name}' for FastRestart and state saving...")
        eng.load_system(self.model_name, nargout=0)
        
        # 1. Turn FastRestart off to set persistent parameters
        eng.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
        
        # 2. Configure how the final operating point is saved
        eng.set_param(self.model_name,
                      'SaveFinalState', 'on',
                      'FinalStateName', self.OPERATING_POINT_VAR_NAME, # Save to this MATLAB var
                      'SaveOperatingPoint', 'on', # Ensures it's a ModelOperatingPoint object
                      'LoadInitialState', 'off', # Model default for LoadInitialState
                      'InitialState', '[]',      # Model default for InitialState
                      nargout=0)
        print(f"Model configured to save final operating point to MATLAB variable '{self.OPERATING_POINT_VAR_NAME}'.")

        # 3. Turn FastRestart on for the session
        eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
        print(f"Model '{self.model_name}' FastRestart is ON.")
        
        # _first_step is managed by reset() setting self.last_operating_point to None

        max_force = 10.0 # Define your action space
        self.action_space = spaces.Box(
            low=-max_force, high=+max_force,
            shape=(1,), dtype=np.float32
        )
        # Define your observation space (e.g., [angle, angular_velocity])
        obs_high_bounds = np.array([np.pi, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high_bounds, high=obs_high_bounds, dtype=np.float32
        )

        # Initial parameter settings (noise, etc.) - these can be set once
        # if they don't change per episode, or in reset() if they do.
        # For now, let's assume they are set once here.
        # Ensure block paths and parameter names are correct for your model.
        try:
            print("Setting initial model parameters (angle, noise)...")
            initial_angle_val = np.random.uniform(-0.1, 0.1) # Small initial angle perturbation
            # Assuming 'initAngleParam' is the name of the parameter for initial angle
            eng.set_param(f'{self.model_name}/Pendulum and Cart', 'initAngle', str(initial_angle_val), nargout=0)

            noise_seed_val  = str(np.random.randint(1, high=40000))
            noise_power_val = '0' # Assuming '0' means no noise, adjust if needed
            eng.set_param(f'{self.model_name}/Noise',   'Seed', noise_seed_val, nargout=0)
            eng.set_param(f'{self.model_name}/Noise',   'Variance',  noise_power_val,  nargout=0)
            noise_seed_v_val = str(np.random.randint(1, high=40000))
            eng.set_param(f'{self.model_name}/Noise_v', 'Seed', noise_seed_v_val, nargout=0)
            eng.set_param(f'{self.model_name}/Noise_v', 'Variance', noise_power_val, nargout=0)
            print("Initial model parameters set.")
        except Exception as e:
            print(f"WARNING: Could not set some initial model parameters: {e}")
            print("Please ensure block paths and parameter names for angle and noise are correct in __init__.")


    def get_data(self):
        """
        Collects and returns the workspace data from 'out.angle' and 'out.tout'.
        (This is the original get_data from the user, with minor parsing improvements)
        """
        angle_lst = []
        time_lst = []
        try:
            # Access elements from the 'out' struct in MATLAB workspace
            # Ensure your Simulink model logs 'angle' and 'tout' to a variable named 'out'
            # (e.g., using "To Workspace" blocks or output ports configuration)
            raw_ang  = eng.workspace['out'].angle
            raw_time = eng.workspace['out'].tout

            # Handle MATLAB's data types (often matlab.double arrays)
            if isinstance(raw_ang, float): # Single value
                angle_lst = [raw_ang]
            elif isinstance(raw_ang, matlab.double) and raw_ang.size == (1,0) : # Empty matlab array
                angle_lst = []
                print("Warning: get_data received empty 'out.angle'.")
            else: # Should be a list or matlab.double array
                angle_lst = [item[0] for item in raw_ang] if raw_ang else []


            if isinstance(raw_time, float):
                time_lst = [raw_time]
            elif isinstance(raw_time, matlab.double) and raw_time.size == (1,0) : # Empty matlab array
                time_lst = []
                print("Warning: get_data received empty 'out.tout'.")
            else:
                time_lst = [item[0] for item in raw_time] if raw_time else []
        
        except Exception as e:
            print(f"Error in get_data accessing 'out.angle' or 'out.tout': {e}")
            print("Ensure Simulink model logs 'angle' and 'tout' to 'out' struct in base workspace.")
            return [], [] # Return empty lists on error

        # print(f"get_data angle_lst: {angle_lst}") # Can be verbose
        # print(f"get_data time_lst: {time_lst}")
        return angle_lst, time_lst

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # For Gym API compatibility
        print("\nResetting environment...")
        self.current_time = 0.0
        self.last_operating_point = None # Crucial: signifies first step after reset

        # Stop any ongoing simulation (good practice, though FastRestart handles some of this)
        try:
            eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        except Exception as e:
            print(f"Note: Error sending 'stop' command during reset (often benign): {e}")

        # Set random initial conditions for the model if they change per episode
        # Example: Randomize initial pendulum angle parameter
        try:
            initial_angle_val = np.random.uniform(-0.1, 0.1) # e.g. +/- ~5.7 degrees
            while -0.01 < initial_angle_val < 0.01: # Avoid perfectly zero for more challenge
                 initial_angle_val = np.random.uniform(-0.1, 0.1)
            # Assuming 'initAngle' is the parameter name in your 'Pendulum and Cart' block/subsystem
            eng.set_param(f'{self.model_name}/Pendulum and Cart', 'initAngle', str(initial_angle_val), nargout=0)
            print(f"Reset: Initial angle parameter set to {initial_angle_val:.3f}")
        except Exception as e:
            print(f"Warning: Could not set initial angle parameter during reset: {e}")


        # Perform an initial simulation run for StopTime '0'
        # This run uses the model's current (randomized) initial conditions.
        # It does NOT load any prior operating point.
        # It WILL save its final operating point to self.OPERATING_POINT_VAR_NAME.
        print(f"Reset: Running initial sim (StopTime '0', LoadInitialState 'off') to establish '{self.OPERATING_POINT_VAR_NAME}'...")
        try:
            # FastRestart is already ON. Parameters for saving state are already set on the model.
            # The sim command itself doesn't need to repeat them when FastRestart is on.
            # We need to ensure 'out' is populated.
            # LoadInitialState is 'off' by virtue of not specifying it and relying on model's default,
            # or by explicitly setting it on the model (which we did in __init__).
            # For this very first segment, ensure we don't load an old op point.
            eng.workspace['sim_options'] = eng.simset('LoadInitialState', 'off')
            command = (
                f"out = sim('{self.model_name}', "
                f"'StopTime', num2str({self.current_time}), " # Effectively StopTime '0'
                f"'LoadInitialState', 'off');" # Override model default if necessary for this call
            )
            # print(f"Reset sim command: {command}")
            eng.eval(command, nargout=0)
            
            # Store the resulting operating point for the first actual step
            self.last_operating_point = eng.workspace[self.OPERATING_POINT_VAR_NAME]
            print(f"Reset: Initial operating point '{self.OPERATING_POINT_VAR_NAME}' captured.")

        except Exception as e:
            print(f"CRITICAL ERROR during initial simulation in reset: {e}")
            # This is a critical failure; the environment might not be usable.
            # Return a default observation and mark as unstable.
            return np.array([0.0, 0.0], dtype=np.float32), {"error": "Reset sim failed"}

        # Get observation from this initial state
        angle_lst, time_lst = self.get_data()
        if not angle_lst: # Check if data is empty
            theta0 = initial_angle_val # Fallback to parameter if sim output is missing
            print(f"Warning: No angle data from initial reset sim, using param value {theta0:.3f}")
        else:
            theta0 = angle_lst[-1]
        
        vel0 = 0.0 # Velocity is typically zero at t=0 after reset if starting from rest.
                   # If get_data provides enough points, you could calculate it.
        if len(angle_lst) >= 2 and len(time_lst) >= 2:
            dt_val = time_lst[-1] - time_lst[-2] if len(time_lst) >=2 else self.dt
            if abs(dt_val) > 1e-9 : # Avoid division by zero
                vel0 = (angle_lst[-1] - angle_lst[-2]) / dt_val
            else: vel0 = 0.0
        
        obs = np.array([theta0, vel0], dtype=np.float32)
        print(f"Reset: Initial observation: [{theta0:.3f}, {vel0:.3f}]")
        return obs, {} # Return observation and empty info dict for SB3

    def step(self, action):
        # 1. Clip & apply action
        # Ensure action is scalar if your Simulink 'Constant' block expects a scalar
        u_scalar = float(action[0]) if isinstance(action, (np.ndarray, list)) and len(action)>0 else float(action)
        u_clipped = np.clip(u_scalar, self.action_space.low[0], self.action_space.high[0])
        
        try:
            # Assuming 'Constant' block at path self.agent_block_path takes the action
            eng.set_param(self.agent_block_path, 'Value', str(u_clipped), nargout=0)
        except Exception as e:
            print(f"CRITICAL ERROR setting action in Simulink: {e}")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            # Return current observation, high penalty, terminated, truncated, info
            return obs, -200, True, False, {"error": "Failed to set action"}

        # 2. Determine simulation stop time for this step
        target_stop_time = self.current_time + self.dt

        # 3. Prepare and run simulation
        # FastRestart is ON. SaveFinalState & FinalStateName are already configured on the model.
        # We need to load the InitialState from self.last_operating_point.
        sim_command_parts = [
            f"out = sim('{self.model_name}',",
            f"'StopTime', num2str({target_stop_time:.10g})" # Use enough precision for time
        ]

        if self.last_operating_point is not None:
            # Pass the MATLAB structure/object directly as InitialState
            # Put the op point into a temporary workspace variable for sim command
            eng.workspace['currentOpStateForSim'] = self.last_operating_point
            sim_command_parts.append(",'LoadInitialState', 'on', 'InitialState', 'currentOpStateForSim'")
            # print(f"Step: Loading InitialState from previous {self.OPERATING_POINT_VAR_NAME}")
        else:
            # This case should ideally be handled by reset ensuring last_operating_point is set.
            # If it's None, it implies this is the very first step after a reset that didn't capture state,
            # or an error occurred. We'll run with LoadInitialState 'off'.
            sim_command_parts.append(",'LoadInitialState', 'off'")
            print(f"Warning: Step {self.current_time/self.dt if self.dt else 0:.0f}: last_operating_point is None. Running with LoadInitialState 'off'.")

        sim_command_parts.append(");")
        full_sim_command = "".join(sim_command_parts)
        # print(f"Step sim command: {full_sim_command}") # For debugging

        try:
            eng.eval(full_sim_command, nargout=0)
            # After successful sim, capture the new operating point
            self.last_operating_point = eng.workspace[self.OPERATING_POINT_VAR_NAME]
        except Exception as e:
            print(f"CRITICAL ERROR during step simulation: {e}")
            print(f"Failed command: {full_sim_command}")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -250, True, False, {"error": f"Simulink sim command failed: {e}"}

        # 4. Get data from simulation output
        angle_lst, time_lst = self.get_data()

        if not angle_lst or not time_lst:
            print(f"Error: Step failed to get data. TargetTime={target_stop_time:.3f}")
            obs = np.zeros(self.observation_space.shape, dtype=np.float32) # Fallback
            # If data is missing, it might be that the simulation didn't run as expected.
            # Try to use last known valid state or terminate.
            # For now, return a penalty and terminate.
            self.current_time = target_stop_time # Advance time anyway or mark as error
            return obs, -300, True, False, {"error": "Missing data after step sim"}

        theta = angle_lst[-1]
        actual_sim_time = time_lst[-1] # Time reported by Simulink

        # 5. Calculate velocity
        vel = 0.0
        if len(angle_lst) >= 2 and len(time_lst) >= 2:
            dt_calc = time_lst[-1] - time_lst[-2]
            if abs(dt_calc) > 1e-9: # Avoid division by zero if dt_calc is effectively 0
                vel = (angle_lst[-1] - angle_lst[-2]) / dt_calc
            # If dt_calc is zero, vel remains 0.0 which is a safe default.
        
        # Ensure theta and vel are scalar if they came as single-element lists from get_data
        if isinstance(theta, list): theta = theta[0] if theta else 0.0
        if isinstance(vel, list): vel = vel[0] if vel else 0.0

        obs = np.array([theta, vel], dtype=np.float32)
        
        # 6. Calculate reward
        # Example: encourage upright (cos(theta) near 1), penalize large control effort and velocity
        reward = np.cos(theta) - 0.001 * (u_clipped**2) - 0.01 * (vel**2)

        # 7. Determine termination and truncation
        self.current_time = actual_sim_time # Update agent's view of time

        terminated = bool(abs(theta) > self.angle_threshold)
        truncated = bool(self.current_time >= self.max_episode_time - self.dt/2) # Buffer for float comparison

        if terminated:
            reward -= 10 # Additional penalty for falling
            # print(f"Terminated: Angle {theta:.3f} exceeded threshold at t={self.current_time:.3f}")
        # if truncated:
            # print(f"Truncated: Max episode time reached at t={self.current_time:.3f}")
            
        return obs, reward, terminated, truncated, {"time": self.current_time}

    def render(self, mode='human'):
        # Typically, Simulink scopes are used for rendering.
        # This method can be used to trigger scope updates if controllable via MATLAB commands.
        pass

    def close(self):
        global eng
        if eng is not None:
            print("Closing Simulink environment...")
            try:
                # It's good practice to turn FastRestart off if you set it on
                eng.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
                print(f"Model '{self.model_name}' FastRestart turned OFF.")
                # eng.close_system(self.model_name, 0, nargout=0) # Optionally close the model
            except Exception as e:
                print(f"Note: Error during model cleanup (e.g., turning off FastRestart): {e}")
            
            try:
                print("Quitting MATLAB engine...")
                eng.quit()
            except Exception as e:
                print(f"Error quitting MATLAB engine: {e}")
            finally:
                eng = None # Clear the global reference
        print("Environment closed.")