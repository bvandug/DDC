"""Gymnasium environment for running a Simulink-based pendulum model.

Spawns a dedicated MATLAB engine, creates a unique model copy for
isolation, steps the simulation in sync with Gymnasium, and supports
seeded reproducibility and optional observation noise.
"""
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import jax
import shutil
import tempfile
import uuid
import os
import shutil
# >>> ADDED FOR DQN --------------------------------------------------------------
class DiscretizedActionWrapper(gym.ActionWrapper):
    """Map discrete action indices to predefined continuous force values.

    Intended for training DQN on a continuous-force Simulink model
    by converting discrete indices to a small set of torque/force levels.

    Args:
        env (gym.Env): Environment to wrap.
        force_values (array-like): Discrete force values to allow.

    Attributes:
        force_values (np.ndarray): Allowed force levels.
        action_space (spaces.Discrete): Discrete action space [0..n-1].
    """

    def __init__(self, env, force_values):
        super().__init__(env)
        self.force_values = np.asarray(force_values, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.force_values))

    def action(self, act_idx):
        """Convert a discrete index into a 1D array containing the force value.

        Args:
            act_idx (int): Index of the chosen discrete action.

        Returns:
            np.ndarray: Continuous force array with shape (1,).
        """
        return np.array([self.force_values[int(act_idx)]], dtype=np.float32)
# ------------------------------------------------------------------------------


class SimulinkEnv(gym.Env):
    """Custom Gymnasium environment backed by a Simulink model.

    Creates a unique copy of the Simulink `.slx` file, runs in FastRestart
    mode for speed, and interacts with MATLAB engine calls for state I/O.

    Args:
        model_name (str): Base name of Simulink model (without extension).
        dt (float): Simulation time step in seconds.
        max_episode_time (float): Episode time limit (s).
        angle_threshold (float): Termination angle threshold (rad).
        seed (int, optional): Seed for RNG initialization.
        eval_obs_noise_std (float): Std. dev. of Gaussian observation noise.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        model_name: str = "pendulum",
        dt: float = 0.01,
        max_episode_time: float = 5,
        angle_threshold: float = np.pi / 2,
        seed: int = None,  # Add seed parameter
        eval_obs_noise_std: float = 0.0,   # single scalar σ
    ):
        super().__init__()

        self._pending_reset_timings = None #for timing from reset method- to flush in step

        # Add JAX-style seeding to match your JAX implementation exactly
        self.rng = jax.random.PRNGKey(seed if seed is not None else 0)
        self.obs_noise_std = float(eval_obs_noise_std)        # NEW
        self.np_rng = np.random.RandomState(int(seed or 0))   # NEW: RNG for noise

        # Instance-specific MATLAB engine
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab("-nodesktop -licmode onlinelicensing")

        # Create a unique copy of the model file
        unique_id = uuid.uuid4().hex[:8]
        self.model_name = f"{model_name}_{unique_id}"
        self.model_path = os.path.join(tempfile.gettempdir(), f"{self.model_name}.slx")
        shutil.copy(f"{model_name}.slx", self.model_path)

        # Load the unique model copy
        self.eng.load_system(self.model_path, nargout=0)
        self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

        self.dt = dt
        self.current_time = 0.0
        self.max_episode_time = max_episode_time
        self.angle_threshold = angle_threshold
        self.pendulum_length = 0.15

        max_torque = 2.0  # same as ip_jax.PendulumConfig.max_torque
        self.action_space = spaces.Box(
            low=-max_torque, high=+max_torque, shape=(1,), dtype=np.float32
        )

        high = np.array([np.pi, np.finfo(np.float32).max], np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Generate initial angle using the seeded RNG
        self.rng, subkey = jax.random.split(self.rng)
        initial_angle = float(jax.random.uniform(subkey, minval=-1.0, maxval=1.0))
        # Match JAX logic exactly: if abs(angle) < 0.05, add 0.1
        initial_angle = (
            initial_angle + 0.1 if abs(initial_angle) < 0.05 else initial_angle
        )

        # print(initial_angle)


        self.eng.set_param(
            f"{self.model_name}/Pendulum and Cart",
            "init",
            str(initial_angle),
            nargout=0,
        )

    def get_data(self):
        """Retrieve latest angle, angular velocity, and time from Simulink.

        Returns:
            tuple[list, list, list]: (angles, velocities, times) as Python lists.
        """
        raw_ang = self.eng.eval("out.angle", nargout=1)
        raw_vel = self.eng.eval("out.angle_v", nargout=1)
        raw_time = self.eng.eval("out.tout", nargout=1)

        # flatten
        ang2d = [[raw_ang]] if isinstance(raw_ang, float) else raw_ang
        vel2d = [[raw_vel]] if isinstance(raw_vel, float) else raw_vel
        t2d = [[raw_time]] if isinstance(raw_time, float) else raw_time

        angle_lst = [a[0] for a in ang2d]
        vel_lst = [v[0] for v in vel2d]
        time_lst = [t[0] for t in t2d]
        return angle_lst, vel_lst, time_lst

    def reset(self, *, seed=None, options=None):
        """Reset the Simulink model to a fresh initial state (Python timing only)."""

        t_total_start = time.perf_counter()

        self.current_time = 0.0
        if seed is not None:
            self.rng = jax.random.PRNGKey(int(seed))
            self.np_rng = np.random.RandomState(int(seed))

        timings = {}

        # Stop sim
        t0 = time.perf_counter()
        self.eng.set_param(self.model_name, "SimulationCommand", "stop", nargout=0)
        timings["reset/stop_sim"] = time.perf_counter() - t0

        # Clear xFinal
        t0 = time.perf_counter()
        try:
            self.eng.eval("clear xFinal", nargout=0)
        except Exception:
            pass
        timings["reset/clear_xFinal"] = time.perf_counter() - t0

        # New initial angle (same logic)
        self.rng, subkey = jax.random.split(self.rng)
        initial_angle = float(jax.random.uniform(subkey, minval=-1.0, maxval=1.0))
        initial_angle = (initial_angle + 0.1 if abs(initial_angle) < 0.05 else initial_angle)
        # print(initial_angle)

        # Set init param
        t0 = time.perf_counter()
        self.eng.set_param(f"{self.model_name}/Pendulum and Cart", "init", str(initial_angle), nargout=0)
        timings["reset/set_init_set_param"] = time.perf_counter() - t0

        # FastRestart OFF
        t0 = time.perf_counter()
        self.eng.set_param(self.model_name, "FastRestart", "off", "LoadInitialState", "off", nargout=0)
        timings["reset/fast_off"] = time.perf_counter() - t0

        # --- in reset(): warm-up sim ---
        t0 = time.perf_counter()
        # run the MATLAB statements as side-effects (no return value)
        self.eng.eval(
            f"t0=tic; out = sim('{self.model_name}', 'StopTime','1e-4', "
            f"'SaveFinalState','on','StateSaveName','xFinal'); "
            f"xFinal = out.xFinal;",  # keep xFinal in base workspace
            nargout=0,
        )
        # now retrieve the elapsed time as the *result* of a single expression
        elapsed_matlab = self.eng.eval("toc(t0)", nargout=1)
        timings["reset/sim_warmup_py"] = time.perf_counter() - t0
        timings["reset/sim_warmup_matlab"] = float(elapsed_matlab)



        # FastRestart ON
        t0 = time.perf_counter()
        self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)
        timings["reset/fast_on"] = time.perf_counter() - t0

        # Read outputs (and time each eval)
        t0 = time.perf_counter()
        t1 = time.perf_counter(); raw_ang  = self.eng.eval("out.angle",  nargout=1); timings["eval_out_angle"] = time.perf_counter() - t1
        t1 = time.perf_counter(); raw_vel  = self.eng.eval("out.angle_v", nargout=1); timings["eval_out_vel"]   = time.perf_counter() - t1
        t1 = time.perf_counter(); raw_time = self.eng.eval("out.tout",    nargout=1); timings["eval_out_time"]  = time.perf_counter() - t1
        timings["reset/get_data"] = time.perf_counter() - t0

        # Flatten and build obs (unchanged)
        ang2d = [[raw_ang]] if isinstance(raw_ang, float) else raw_ang
        vel2d = [[raw_vel]] if isinstance(raw_vel, float) else raw_vel
        t2d   = [[raw_time]] if isinstance(raw_time, float) else raw_time
        angle_lst = [a[0] for a in ang2d]
        vel_lst   = [v[0] for v in vel2d]
        time_lst  = [t[0] for t in t2d]

        theta = angle_lst[-1]
        vel   = vel_lst[-1]
        t     = time_lst[-1]
        obs = np.array([theta, vel], dtype=np.float32)
        if self.obs_noise_std > 0.0:
            obs += self.np_rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)

        info = {"time": float(t), "timings": timings}
        # After all timings have been filled in
        timings["reset/total_reset_time"] = time.perf_counter() - t_total_start

        matlab_wall_keys = [
            "reset/stop_sim","reset/clear_xFinal","reset/set_init_set_param",
            "reset/fast_off","reset/sim_warmup_py","reset/fast_on",
            "reset/get_data",  # keep group
            # (omit eval_out_angle/vel/time to avoid double-count)
        ]

        total_matlab_wall = sum(timings.get(k, 0.0) for k in matlab_wall_keys)
        total_matlab_compute = timings.get("reset/sim_warmup_matlab", 0.0)
        total_python_time = timings["reset/total_reset_time"] - total_matlab_wall

        timings["reset/total_matlab_wall"] = total_matlab_wall
        timings["reset/total_matlab_compute"] = total_matlab_compute
        timings["reset/total_python_time"] = total_python_time

        self._pending_reset_timings = timings.copy()
        return obs, info



    def step(self, action):
        """Advance the Simulink model by one dt (Python timing only)."""

        timings = {}

        if self._pending_reset_timings is not None:
            for k, v in self._pending_reset_timings.items():
                timings.setdefault(k, v)  # keep "reset/..." keys as-is
            self._pending_reset_timings = None

        t_total_start = time.perf_counter()
       
        torque = float(np.clip(action, self.action_space.low, self.action_space.high))

        # set_param for action
        t0 = time.perf_counter()
        self.eng.set_param(f"{self.model_name}/Constant", "Value", str(torque), nargout=0)
        timings["step/set_action_set_param"] = time.perf_counter() - t0

        # FastRestart OFF
        t0 = time.perf_counter()
        self.eng.set_param(self.model_name, "FastRestart", "off", nargout=0)
        timings["step/fast_off"] = time.perf_counter() - t0

        # Simulate one step: add MATLAB-side timing
        stop = self.current_time + self.dt

        # --- in step(): one simulation step ---
        t0 = time.perf_counter()
        self.eng.eval(
            f"t0=tic; out = sim('{self.model_name}', "
            f"'LoadInitialState','on','InitialState','xFinal', "
            f"'StopTime','{stop}', 'SaveFinalState','on','StateSaveName','xFinal'); "
            f"xFinal = out.xFinal;",
            nargout=0,
        )
        elapsed_matlab = self.eng.eval("toc(t0)", nargout=1)
        timings["step/sim_py"] = time.perf_counter() - t0
        timings["step/sim_matlab"] = float(elapsed_matlab)

        # FastRestart ON
        t0 = time.perf_counter()
        self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)
        timings["step/fast_on"] = time.perf_counter() - t0

        # Read outputs (time each eval)
        t0 = time.perf_counter()
        t1 = time.perf_counter(); raw_ang  = self.eng.eval("out.angle",  nargout=1); timings["eval_out_angle"] = time.perf_counter() - t1
        t1 = time.perf_counter(); raw_vel  = self.eng.eval("out.angle_v", nargout=1); timings["eval_out_vel"]   = time.perf_counter() - t1
        t1 = time.perf_counter(); raw_time = self.eng.eval("out.tout",    nargout=1); timings["eval_out_time"]  = time.perf_counter() - t1
        timings["step/get_data"] = time.perf_counter() - t0

        # Build obs + reward (unchanged)
        ang2d = [[raw_ang]] if isinstance(raw_ang, float) else raw_ang
        vel2d = [[raw_vel]] if isinstance(raw_vel, float) else raw_vel
        t2d   = [[raw_time]] if isinstance(raw_time, float) else raw_time
        angle_lst = [a[0] for a in ang2d]
        vel_lst   = [v[0] for v in vel2d]
        time_lst  = [t[0] for t in t2d]

        theta = angle_lst[-1]
        vel   = vel_lst[-1]
        t     = time_lst[-1]
        obs = np.array([theta, vel], dtype=np.float32)
        if self.obs_noise_std > 0.0:
            obs += self.np_rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)

        reward = float(np.cos(theta))
        terminated = bool(abs(theta) > self.angle_threshold)   # episode end by condition
        truncated  = bool(t >= self.max_episode_time)          # episode end by time limit
        self.current_time = t

        info = {"time": t, "timings": timings}
        timings["step/total_step_time"] = time.perf_counter() - t_total_start

        # Everything that touches the engine this step (wall-time view)
        matlab_wall_keys = [
            "step/set_action_set_param",
            "step/fast_off",
            "step/sim_py",
            "step/fast_on",
            "step/get_data",  # keep group
            # (omit eval_out_angle/vel/time to avoid double-count)
        ]

        total_matlab_wall = sum(timings.get(k, 0.0) for k in matlab_wall_keys)

        # True MATLAB compute (if recorded)
        total_matlab_compute = timings.get("step/sim_matlab", 0.0)

        # Everything else is Python-only work inside step()
        total_python_time = timings["step/total_step_time"] - total_matlab_wall

        timings["step/total_matlab_wall"] = total_matlab_wall
        timings["step/total_matlab_compute"] = total_matlab_compute
        timings["step/total_python_time"] = total_python_time


        return obs, reward, terminated, truncated, info


    def render(self, mode="human"):
        """Render the environment (currently not implemented)."""
        pass

    def close(self):
        """Shut down MATLAB engine and clean up temporary files.

        Stops the Simulink engine session, deletes the temporary model copy,
        and removes cached build folders and autosave files.
        """
        self.eng.quit()

        # Clean up temporary model file
        if hasattr(self, "model_path") and os.path.exists(self.model_path):
            try:
                os.remove(self.model_path)
                print(f"Deleted temporary model file: {self.model_path}")
            except Exception as e:
                print(f"Warning: could not delete model file: {e}")

        # Clean slprj/<model_name> folder (worker-specific)
        slprj_model_dir = os.path.join(os.getcwd(), "slprj", self.model_name)
        if os.path.exists(slprj_model_dir):
            try:
                shutil.rmtree(slprj_model_dir)
                print(f"Deleted slprj cache for model: {self.model_name}")
            except Exception as e:
                print(f"Warning: could not delete slprj folder: {e}")

        # Remove any autosave or .slxc file linked to this model
        base_name = os.path.splitext(self.model_name)[0]
        autosave_file = f"{base_name}.slx.autosave"
        slxc_file = f"{base_name}.slxc"

        for filename in [autosave_file, slxc_file]:
            full_path = os.path.join(os.getcwd(), filename) 
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    print(f"Deleted: {full_path}")
                except Exception as e:
                    print(f"Warning: could not delete file: {full_path}: {e}")