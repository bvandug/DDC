import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matlab.engine
import time
import traceback # For detailed error printing

SIM_TIME_PRECISION = 6 
# Name of the variable Simulink will save the Outport data structure to
# (must match Model Settings if "Single simulation output" is UNCHECKED)
SIMULINK_ACTUAL_YOUT_VARIABLE_NAME = 'actual_yout_data' 

class CustomPendulumEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, model_name, matlab_engine_instance, dt=0.02, max_steps_per_episode=250):
        super().__init__()
        self.model_name = model_name
        self.dt = dt
        self.max_steps_per_episode = max_steps_per_episode
        self.eng = matlab_engine_instance
        
        self.force_input_block_path = f"{self.model_name}/RLForceInput"
        self.initial_angle_subsystem_path = f"{self.model_name}/Pendulum and Cart"
        self.initial_angle_param_name = "init"
        
        self.current_rl_step = 0
        self.current_sim_time = 0.0

        self.force_limit = 10.0
        self.action_space = spaces.Box(low=-self.force_limit, high=self.force_limit, shape=(1,), dtype=np.float32)
        self.angle_limit = np.pi
        self.angular_velocity_limit = 15.0
        self.observation_space = spaces.Box(
            low=np.array([-self.angle_limit, -self.angular_velocity_limit]),
            high=np.array([self.angle_limit, self.angular_velocity_limit]),
            dtype=np.float32
        )
        self.fall_angle_threshold = np.pi / 2.0

        try:
            print(f"Using provided MATLAB engine. Loading Simulink model '{self.model_name}'...")
            self.eng.load_system(self.model_name, nargout=0)
            print(f"Simulink model '{self.model_name}' loaded.")
            current_fr_setting = self.eng.get_param(self.model_name, 'FastRestart')
            print(f"Model '{self.model_name}' current FastRestart setting from file: {current_fr_setting}")
            if current_fr_setting.lower() != 'on':
                print("Warning: FastRestart is not 'on' in the loaded model. Attempting to enable...")
                self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
                print(f"FastRestart set to 'on'. Current: {self.eng.get_param(self.model_name, 'FastRestart')}")
        except Exception as e:
            print(f"Error during Simulink model initialization with provided engine: {e}")
            raise
        self.last_action_applied = 0.0

    def _get_observation_from_sim_output(self, directly_fetched_yout_struct):
        # This method now receives the 'actual_yout_data' structure directly.
        # It no longer needs to check for ErrorMessage first, as that would be part of a
        # different handling if the sim command itself errored.
        # We assume 'directly_fetched_yout_struct' IS the structure containing 'signals'.
        try:
            if directly_fetched_yout_struct is None:
                print("CRITICAL Error in _get_observation_from_sim_output: received None for Outport data structure.")
                return np.array([0.0, 0.0], dtype=np.float32)

            angle, angle_v = None, None
            signals_container = None
            
            # directly_fetched_yout_struct should be the object that has the .signals field
            # This corresponds to the 'yout' field from the SimulationOutput object,
            # or the variable itself if "Single simulation output" is off.
            if isinstance(directly_fetched_yout_struct, matlab.object):
                if hasattr(directly_fetched_yout_struct, 'signals'):
                    signals_container = directly_fetched_yout_struct.signals
                else:
                    print(f"Debug: directly_fetched_yout_struct (matlab.object) does NOT have 'signals' attribute. Fields: {self.eng.fieldnames(directly_fetched_yout_struct) if hasattr(self.eng,'fieldnames') and hasattr(directly_fetched_yout_struct,'size') else 'Cannot get fieldnames'}")
            elif isinstance(directly_fetched_yout_struct, dict):
                if 'signals' in directly_fetched_yout_struct:
                     signals_container = directly_fetched_yout_struct['signals']
                else:
                    print(f"Debug: directly_fetched_yout_struct (dict) does NOT have 'signals' key. Keys: {directly_fetched_yout_struct.keys()}")
            else:
                print(f"Debug: directly_fetched_yout_struct is not a recognized type for parsing signals. Type: {type(directly_fetched_yout_struct)}")


            if signals_container is not None:
                num_signals = 0
                is_matlab_array = isinstance(signals_container, matlab.object) and hasattr(signals_container, 'size')
                
                if is_matlab_array: 
                    size_info = signals_container.size
                    if isinstance(size_info, (list, tuple)) and len(size_info) > 0:
                        if size_info[0] > 0 and (len(size_info) == 1 or (len(size_info) > 1 and size_info[1] > 0)) : 
                            num_signals = size_info[0] 
                    elif isinstance(size_info, (int, float)): num_signals = int(size_info)
                elif hasattr(signals_container, '__len__'): 
                    num_signals = len(signals_container)
                
                if num_signals == 0 and signals_container is not None: 
                    num_signals = 1

                for i in range(num_signals):
                    sig_struct = None
                    try: 
                        sig_struct = signals_container[i] if num_signals > 1 else signals_container
                    except Exception as e_idx:
                        print(f"Debug: Error indexing signals_container at index {i}: {e_idx}"); continue
                    
                    if sig_struct is not None:
                        label = str(getattr(sig_struct, 'label', ''))
                        current_value = None
                        values_field = getattr(sig_struct, 'values', None)
                        if values_field is not None:
                            if isinstance(values_field, matlab.object) and values_field.size > 0:
                                val_data = values_field[-1] 
                                current_value = float(val_data[0] if isinstance(val_data, matlab.object) and val_data.size > 0 else val_data)
                            elif isinstance(values_field, (list, np.ndarray)) and len(values_field) > 0: current_value = float(values_field[-1])
                            elif isinstance(values_field, (float, int, np.number)): current_value = float(values_field)
                        
                        if current_value is not None:
                            if label == 'angle_out_signal' or label == 'angle_out': angle = current_value
                            elif label == 'angle_v_out_signal' or label == 'angle_v_out': angle_v = current_value
                            elif angle is None and ('angle' in label and 'v_out_signal' not in label and '_v' not in label and 'velocity' not in label): angle = current_value
                            elif angle_v is None and ('angle_v' in label or 'ang_vel' in label or ('angle' in label and '_v' in label)): angle_v = current_value
            
            if angle is not None and angle_v is not None:
                # print(f"Successful parse: angle={angle}, angle_v={angle_v}")
                return np.array([angle, angle_v], dtype=np.float32)
            else:
                print(f"Warning: Could not parse angle (found: {angle}) and/or angle_v (found: {angle_v}) from the provided Outport data structure.")
                if directly_fetched_yout_struct is not None:
                    print(f"Debug (Final Attempt): directly_fetched_yout_struct type: {type(directly_fetched_yout_struct)}")
                    if isinstance(directly_fetched_yout_struct, matlab.object):
                         try: print(f"Debug (Final Attempt): directly_fetched_yout_struct fields: {self.eng.fieldnames(directly_fetched_yout_struct)}")
                         except: print("Debug (Final Attempt): Could not get fieldnames for directly_fetched_yout_struct.")
                    elif isinstance(directly_fetched_yout_struct, dict): print(f"Debug (Final Attempt): directly_fetched_yout_struct keys: {directly_fetched_yout_struct.keys()}")
                
                print("CRITICAL: Failed to retrieve valid observation values from Outport data structure.")
                return np.array([0.0, 0.0], dtype=np.float32)
        except Exception as e:
            print(f"CRITICAL Exception at outer level of _get_observation_from_sim_output: {type(e).__name__} - {e}")
            traceback.print_exc()
            return np.array([0.0, 0.0], dtype=np.float32)

    def _calculate_reward(self, obs):
        angle = obs[0]; reward = np.cos(angle); return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_rl_step = 0
        self.current_sim_time = 0.0 
        
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        init_angle_val = self.np_random.uniform(low=-0.1, high=0.1)
        try:
            self.eng.set_param(self.initial_angle_subsystem_path, self.initial_angle_param_name, str(init_angle_val), nargout=0)
        except Exception as e: 
            print(f"CRITICAL ERROR: Failed to set initial angle: {e}")
            return np.array([0.0, 0.0], dtype=np.float32), {"error": "Failed to set initial angle"}

        actual_yout_data_from_sim = None # Initialize to None
        sim_stop_time_str = '0.001' 
        
        try:
            print(f"Reset: Running sim to {sim_stop_time_str}s. Simulink should save Outport data to '{SIMULINK_ACTUAL_YOUT_VARIABLE_NAME}'.")
            sim_command = f"sim('{self.model_name}', 'StopTime', '{sim_stop_time_str}');"
            self.eng.eval(sim_command, nargout=0) # nargout=0, relies on workspace variable
            
            # print(f"Reset: Attempting to fetch '{SIMULINK_ACTUAL_YOUT_VARIABLE_NAME}' from MATLAB workspace...")
            actual_yout_data_from_sim = self.eng.workspace[SIMULINK_ACTUAL_YOUT_VARIABLE_NAME]
            # print(f"Reset: Fetched '{SIMULINK_ACTUAL_YOUT_VARIABLE_NAME}' successfully. Type: {type(actual_yout_data_from_sim)}")

        except matlab.engine.MatlabExecutionError as me_err:
            print(f"MATLAB Execution Error during reset sim or fetch: {me_err}")
        except KeyError:
             print(f"CRITICAL ERROR in reset: '{SIMULINK_ACTUAL_YOUT_VARIABLE_NAME}' not found in MATLAB workspace after simulation.")
        except Exception as e:
            print(f"General Error during reset simulation or fetch: {type(e).__name__} - {e}");
            traceback.print_exc()

        initial_obs = self._get_observation_from_sim_output(actual_yout_data_from_sim) 
        self.current_sim_time = 0.0 
        return initial_obs, {}

    def step(self, action):
        force_applied = float(action[0])
        self.last_action_applied = force_applied
        try:
            self.eng.set_param(self.force_input_block_path, 'Value', str(force_applied), nargout=0)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to set force: {e}")
            obs_err = np.array([0.0,0.0],dtype=np.float32); last_yout_data = None
            try: last_yout_data = self.eng.workspace.get(SIMULINK_ACTUAL_YOUT_VARIABLE_NAME, None)
            except: pass
            if last_yout_data: obs_err = self._get_observation_from_sim_output(last_yout_data)
            return obs_err, -200.0, True, False, {"error": "Simulink communication failed on set_param"}

        target_sim_time = self.current_sim_time + self.dt
        target_sim_time_str = f"{target_sim_time:.{SIM_TIME_PRECISION}f}" 
        
        actual_yout_data_from_sim_step = None
        try:
            sim_command = f"sim('{self.model_name}', 'StopTime', '{target_sim_time_str}');"
            self.eng.eval(sim_command, nargout=0) # nargout=0
            actual_yout_data_from_sim_step = self.eng.workspace[SIMULINK_ACTUAL_YOUT_VARIABLE_NAME]
        except matlab.engine.MatlabExecutionError as me_err:
            print(f"MATLAB Execution Error during step sim or fetch (StopTime: {target_sim_time_str}): {me_err}")
        except KeyError:
             print(f"CRITICAL ERROR in step: '{SIMULINK_ACTUAL_YOUT_VARIABLE_NAME}' not found in MATLAB workspace after simulation (StopTime: {target_sim_time_str}).")
        except Exception as e:
            print(f"General Error during step sim or fetch (StopTime: {target_sim_time_str}): {type(e).__name__} - {e}")
            traceback.print_exc()
        
        self.current_sim_time = target_sim_time 
        self.current_rl_step += 1
        obs = self._get_observation_from_sim_output(actual_yout_data_from_sim_step)
        reward = self._calculate_reward(obs)
        terminated = bool(abs(obs[0]) > self.fall_angle_threshold)
        truncated = bool(self.current_rl_step >= self.max_steps_per_episode)
        if terminated: reward = -100.0 
        return obs, reward, terminated, truncated, {}

    def render(self): 
        pass

    def close(self):
        print(f"Closing CustomPendulumEnv for model '{self.model_name}'. (Shared MATLAB engine will not be quit by this instance)")
        try:
            if hasattr(self, 'eng') and self.eng: 
                self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        except Exception as e:
            pass