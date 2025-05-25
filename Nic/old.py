import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matlab.engine
import time

class CustomPendulumEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, model_name, dt=0.02, max_steps_per_episode=250):
        super().__init__()
        self.model_name = model_name
        self.dt = dt
        self.max_steps_per_episode = max_steps_per_episode

        self.force_input_block_path = f"{self.model_name}/RLForceInput"
        self.initial_angle_subsystem_path = f"{self.model_name}/Pendulum and Cart"
        self.initial_angle_param_name = "init"
        
        # No explicit state variables managed by Python for loading/saving to workspace

        self.current_rl_step = 0
        self.current_sim_time = 0.0 # Python tracks current simulation time

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

        print("Initializing MATLAB engine...")
        try:
            self.eng = matlab.engine.start_matlab()
            print("MATLAB engine started.")
            self.eng.load_system(self.model_name, nargout=0)
            print(f"Simulink model '{self.model_name}' loaded.")
            # Check and rely on model's saved FastRestart setting
            current_fr_setting = self.eng.get_param(self.model_name, 'FastRestart')
            print(f"Model '{self.model_name}' current FastRestart setting from file: {current_fr_setting}")
            if current_fr_setting.lower() != 'on':
                print("Warning: FastRestart is not 'on' in the loaded model. Performance may be impacted.")
                print("Attempting to enable FastRestart programmatically for the session...")
                self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
                print(f"FastRestart set to 'on'. Current: {self.eng.get_param(self.model_name, 'FastRestart')}")


        except Exception as e:
            print(f"Error during MATLAB/Simulink initialization: {e}")
            raise
        self.last_action_applied = 0.0

    def _get_observation_from_sim_output(self, sim_out_struct):
        try:
            if not sim_out_struct:
                print("Warning: sim_out_struct is None in _get_observation.")
                return np.array([0.0, 0.0], dtype=np.float32)

            angle_exists = self.eng.isfield(sim_out_struct, 'angle', nargout=1)
            angle_v_exists = self.eng.isfield(sim_out_struct, 'angle_v', nargout=1)

            if not (angle_exists and angle_v_exists):
                field_names = self.eng.fieldnames(sim_out_struct, nargout=1) if sim_out_struct else "sim_out_struct is None"
                print(f"KeyError: 'angle' or 'angle_v' not in sim_out_struct. Available fields: {field_names}")
                return np.array([0.0, 0.0], dtype=np.float32)

            angle_ts = sim_out_struct['angle'] 
            ang_vel_ts = sim_out_struct['angle_v']
            current_angle, current_angular_velocity = 0.0, 0.0

            if isinstance(angle_ts, matlab.double):
                if angle_ts.size == 0: current_angle = 0.0
                elif isinstance(angle_ts.size, tuple) and len(angle_ts.size) > 0 :
                     current_angle = float(angle_ts[-1][0]) if isinstance(angle_ts[-1], list) and len(angle_ts[-1]) > 0 else float(angle_ts[-1])
                elif isinstance(angle_ts.size, (float, int)) and angle_ts.size >= 1.0 : 
                    current_angle = float(angle_ts[0])
                else: pass
            elif isinstance(angle_ts, float): current_angle = angle_ts
            
            if isinstance(ang_vel_ts, matlab.double):
                if ang_vel_ts.size == 0: current_angular_velocity = 0.0
                elif isinstance(ang_vel_ts.size, tuple) and len(ang_vel_ts.size) > 0:
                    current_angular_velocity = float(ang_vel_ts[-1][0]) if isinstance(ang_vel_ts[-1], list) and len(ang_vel_ts[-1]) > 0 else float(ang_vel_ts[-1])
                elif isinstance(ang_vel_ts.size, (float, int)) and ang_vel_ts.size >= 1.0 :
                    current_angular_velocity = float(ang_vel_ts[0])
                else: pass
            elif isinstance(ang_vel_ts, float): current_angular_velocity = ang_vel_ts
            
            return np.array([current_angle, current_angular_velocity], dtype=np.float32)
        except Exception as e:
            print(f"Generic error retrieving observation: {type(e).__name__} - {e}")
            return np.array([0.0, 0.0], dtype=np.float32)

    def _calculate_reward(self, obs):
        angle = obs[0]; reward = np.cos(angle); return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_rl_step = 0
        self.current_sim_time = 0.0 

        # Stop any previous simulation explicitly
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        # For Fast Restart, setting initial conditions and then running a 0-sec sim
        # "commits" these initial conditions for the next run.
        init_angle_val = self.np_random.uniform(low=-0.1, high=0.1)
        try:
            self.eng.set_param(self.initial_angle_subsystem_path, self.initial_angle_param_name, str(init_angle_val), nargout=0)
            print(f"Reset: Set initial angle to {init_angle_val:.3f}")
        except Exception as e: print(f"CRITICAL ERROR: Failed to set initial angle: {e}"); raise

        # We are not programmatically setting LoadInitialState/SaveFinalState/InitialState/FinalStateName here.
        # We rely purely on FastRestart to maintain state and the model's GUI settings for 'out' struct.
        out_struct_reset = None
        try:
            print(f"Reset: Running 0-second simulation (model: {self.model_name}) to apply initial conditions...")
            # This sim applies the initial angle set by set_param, and because FastRestart is on,
            # the model should be in this state for the first 'step' call.
            # The 'out' struct will contain initial observations.
            out_struct_reset = self.eng.eval(f"sim('{self.model_name}', 'StopTime', '0');", nargout=1)
            print(f"Reset: 0-second sim completed.")
        except Exception as e:
            print(f"Error during 0-second reset simulation: {e}"); raise

        initial_obs = self._get_observation_from_sim_output(out_struct_reset)
        return initial_obs, {}

    def step(self, action):
        force_applied = float(action[0])
        self.last_action_applied = force_applied
        try:
            self.eng.set_param(self.force_input_block_path, 'Value', str(force_applied), nargout=0)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to set force: {e}")
            obs_err = np.array([0.0,0.0],dtype=np.float32); out_s_err = None
            try: out_s_err = self.eng.eval(f"sim('{self.model_name}', 'StopTime', '{self.current_sim_time}');", nargout=1)
            except: pass
            if out_s_err: obs_err = self._get_observation_from_sim_output(out_s_err)
            return obs_err, -200.0, True, False, {"error": "Simulink communication failed"}

        target_sim_time = self.current_sim_time + self.dt # Target for this step
        target_sim_time_str = f"{target_sim_time}"

        # We do not set LoadInitialState/InitialState/SaveFinalState/FinalStateName.
        # We rely on FastRestart. The model continues from its previous state up to the new StopTime.
        out_struct_step = None
        try:
            # print(f"Step: Simulating from {self.current_sim_time} to {target_sim_time_str}") # Verbose
            out_struct_step = self.eng.eval(f"sim('{self.model_name}', 'StopTime', '{target_sim_time_str}');", nargout=1)
        except Exception as e:
            print(f"Error during Simulink step sim (StopTime: {target_sim_time_str}): {e}")
            obs_err = np.array([0.0,0.0],dtype=np.float32); out_s_err = None
            try: out_s_err = self.eng.eval(f"sim('{self.model_name}', 'StopTime', '{self.current_sim_time}');", nargout=1)
            except: pass
            if out_s_err: obs_err = self._get_observation_from_sim_output(out_s_err)
            return obs_err, -200.0, True, False, {"error": "Simulink step failed"}

        self.current_sim_time = target_sim_time # Update Python's track of simulation time
        self.current_rl_step += 1

        obs = self._get_observation_from_sim_output(out_struct_step)
        reward = self._calculate_reward(obs)
        terminated = bool(abs(obs[0]) > self.fall_angle_threshold)
        truncated = bool(self.current_rl_step >= self.max_steps_per_episode)
        if terminated: reward = -100.0
        return obs, reward, terminated, truncated, {}

    def render(self): pass
    def close(self):
        print("Shutting down CustomPendulumEnv...")
        try:
            if hasattr(self, 'eng') and isinstance(self.eng, matlab.engine.MatlabEngine):
                # It's good practice to stop simulation if possible before quitting
                try: self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
                except: pass # Ignore if model already closed or command fails
                self.eng.quit()
                print("MATLAB engine quit.")
        except Exception as e: print(f"Error quitting MATLAB engine: {e}")

if __name__ == '__main__':
    MODEL_FILE_NAME = "PendCart" 
    env = CustomPendulumEnv(model_name=MODEL_FILE_NAME, dt=0.02, max_steps_per_episode=250)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    for i in range(1): 
        obs, info = env.reset()
        done = False; ep_steps = 0; total_ep_reward = 0.0
        print(f"\n--- Test Episode {i+1} ---")
        if obs is not None: print(f"Initial Obs: Angle: {obs[0]:.3f}, AngVel: {obs[1]:.3f}")
        else: print("Initial observation was None.")
        while not done and obs is not None and ep_steps < env.max_steps_per_episode :
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_ep_reward += reward; ep_steps += 1
            if ep_steps % 20 == 0 or done :
                 print(f"Step: {ep_steps}, Action: {action[0]:.2f}, Angle: {obs[0]:.3f}, AngVel: {obs[1]:.3f}, Reward: {reward:.3f}, Done: {done}, Info: {info}")
        print(f"Episode finished after {ep_steps} steps. Total reward: {total_ep_reward:.2f}")
    env.close()
    print("Test complete.")