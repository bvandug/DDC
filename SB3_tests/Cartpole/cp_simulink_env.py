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


class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, force_values):
        """ Initialize the wrapper with a fixed set of continuous actions.

            Parameters
            ----------
            env :
                The base environment with a continuous action space.
            force_values :
                Sequence of scalar control values. Each discrete index maps
                to one element of this sequence.
        """
        super().__init__(env)
        self.force_values = np.asarray(force_values, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.force_values))

    def action(self, act_idx):
        """ Map a discrete index to its continuous control value.

            Parameters
            ----------
            act_idx :
                Discrete action index chosen by the policy.

            Returns
            -------
            np.ndarray
                A (1,) float32 array containing the selected force value.
        """
        return np.array([self.force_values[int(act_idx)]], dtype=np.float32)

class SimulinkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
    self,
    model_name: str = "PendCart",
    dt: float = 0.01,
    max_episode_time: float = 5,
    angle_threshold: float = np.pi / 2,
    seed: int = None,
    eval_obs_noise_std: float = 0.0,
    ):
        """ Start MATLAB, load a unique model copy, and initialize state.

            Parameters
            ----------
            model_name : str, optional
                Base name of the Simulink model (without the temp suffix).
            dt : float, optional
                Simulation step (seconds) advanced on each `step()` call.
            max_episode_time : float, optional
                Episode time limit in seconds (simulation time).
            angle_threshold : float, optional
                Terminal angle magnitude (radians) for failure.
            seed : int, optional
                Seed for both JAX and NumPy RNGs.
            eval_obs_noise_std : float, optional
                Std-dev of zero-mean Gaussian noise added to observations.

            Notes
            -----
            - A unique `.slx` copy is created in a temp dir to avoid clashes
            across multiple env instances.
            - The initial angle is sampled with JAX; if |angle| < 0.05, +0.1
            is added.
        """

        super().__init__()

        self.rng = jax.random.PRNGKey(seed if seed is not None else 0)
        self.obs_noise_std = float(eval_obs_noise_std)      
        self.np_rng = np.random.RandomState(int(seed or 0))   

        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()

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
        self.pendulum_length = 1.0

        max_force = 10.0
        self.action_space = spaces.Box(
            low=-max_force, high=+max_force, shape=(1,), dtype=np.float32
        )
        high = np.array([np.pi, np.finfo(np.float32).max], np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.rng, subkey = jax.random.split(self.rng)
        initial_angle = float(jax.random.uniform(subkey, minval=-1.0, maxval=1.0))
        initial_angle = (initial_angle + 0.1 if abs(initial_angle) < 0.05 else initial_angle)
        self.eng.set_param(f"{self.model_name}/Pendulum and Cart", "init", str(initial_angle), nargout=0)

    def get_data(self):
        """ Fetch angle, angular velocity, and time logs from Simulink.

            Returns
            -------
            tuple[list[float], list[float], list[float]]
                Three lists: angles, angular velocities, and time stamps,
                flattened from MATLAB arrays and ordered by simulation time.

            Notes
            -----
            Assumes the model logs `out.angle`, `out.angle_v`, and `out.tout`.
        """

        raw_ang = self.eng.eval("out.angle", nargout=1)
        raw_vel = self.eng.eval("out.angle_v", nargout=1)
        raw_time = self.eng.eval("out.tout", nargout=1)

        ang2d = [[raw_ang]] if isinstance(raw_ang, float) else raw_ang
        vel2d = [[raw_vel]] if isinstance(raw_vel, float) else raw_vel
        t2d = [[raw_time]] if isinstance(raw_time, float) else raw_time

        angle_lst = [a[0] for a in ang2d]
        vel_lst = [v[0] for v in vel2d]
        time_lst = [t[0] for t in t2d]
        return angle_lst, vel_lst, time_lst

    def reset(self, seed=None, options=None):
        """ Reset the simulation and return the initial observation.

            Parameters
            ----------
            seed : int, optional
                If provided, reseeds both JAX and NumPy RNGs.
            options : dict, optional
                Unused Gymnasium options placeholder.

            Returns
            -------
            tuple[np.ndarray, dict]
                Observation array `[theta, theta_dot]` (float32) and an
                info dict containing `{"time": <float>}`.

            Notes
            -----
            Stops any running sim, clears `xFinal`, sets a new initial angle,
            runs a short warm-up sim to populate `xFinal`, and applies optional
            Gaussian obs noise if configured.
        """

        self.current_time = 0.0
        if seed is not None:
            self.rng = jax.random.PRNGKey(int(seed))
            self.np_rng = np.random.RandomState(int(seed))

        self.eng.set_param(self.model_name, "SimulationCommand", "stop", nargout=0)

         # Clear any previous saved states
        try:
            self.eng.eval("clear xFinal", nargout=0)
        except:
            pass  # No xFinal to clear, which is fine

        # Generate new initial angle using seeded RNG
        self.rng, subkey = jax.random.split(self.rng)
        initial_angle = float(jax.random.uniform(subkey, minval=-1.0, maxval=1.0))
        initial_angle = (
            initial_angle + 0.1 if abs(initial_angle) < 0.05 else initial_angle
        )    
        # initial_angle = float(-0.944298505783081)

        self.eng.set_param(
            f"{self.model_name}/Pendulum and Cart",
            "init",
            str(initial_angle),
            nargout=0,
        )

        self.eng.set_param(
            self.model_name, "FastRestart", "off", "LoadInitialState", "off", nargout=0
        )
        self.eng.eval(
            f"out = sim('{self.model_name}', 'StopTime','1e-4', 'SaveFinalState','on', 'StateSaveName','xFinal'); xFinal = out.xFinal;",
            nargout=0,
        )
        self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

        # Get the actual initial state from simulation
        angle_lst, vel_lst, time_lst = self.get_data()
        theta = angle_lst[-1]
        t = time_lst[-1]
        vel = vel_lst[-1]

        obs = np.array([theta, vel], dtype=np.float32)
        if self.obs_noise_std > 0.0:
            obs += self.np_rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)

        return obs, {"time": float(time_lst[-1])}
    
    def step(self, action):
        """ Advance the Simulink model by `dt` using the given control.

            Parameters
            ----------
            action :
                Continuous control input; clipped to the action space bounds.

            Returns
            -------
            tuple
                `(obs, reward, done, info)` where:
                - `obs` is `[theta, theta_dot]` (float32, with optional noise),
                - `reward` is `cos(theta)` (upright is better),
                - `done` is True if angle exceeds the threshold or time limit,
                - `info` contains `{"time": <float>}`.

            Notes
            -----
            For compatibility with existing callers, this returns a single
            `done` flag rather than Gymnasium's `(terminated, truncated)`.
        """

        u = float(np.clip(action, self.action_space.low, self.action_space.high))

        self.eng.set_param(f"{self.model_name}/Constant", "Value", str(u), nargout=0)

        start, stop = self.current_time, self.current_time + self.dt
        self.eng.set_param(self.model_name, "FastRestart", "off", nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}',"
            f" 'LoadInitialState','on',"
            f" 'InitialState','xFinal',"
            f" 'StopTime','{stop}',"
            f" 'SaveFinalState','on',"
            f" 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0,
        )
        self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

        angle_lst, vel_lst, time_lst = self.get_data()
        theta = angle_lst[-1]
        t = time_lst[-1]
        vel = vel_lst[-1]
        obs = np.array([theta, vel], dtype=np.float32)

        if self.obs_noise_std > 0.0:
            obs += self.np_rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)

        reward = np.cos(theta)
        done = abs(theta) > self.angle_threshold or t >= self.max_episode_time
        terminated = bool(abs(theta) > self.angle_threshold)
        truncated  = bool(t >= self.max_episode_time)        
        self.current_time = t

        return obs, reward, done, {"time": t}


    def render(self, mode="human"):
        """No custom renderer. Integrate Simulink visualization if needed."""

        pass

    def close(self):
        """ Shut down MATLAB and delete temporary model artifacts.

            Closes the MATLAB engine, removes the unique temp `.slx` file,
            cleans any worker-specific `slprj/<model_name>` cache, and deletes
            related autosave/`.slxc` files. Failures are logged as warnings.
        """

        import os
        import shutil
        import glob

        self.eng.quit()

        # Clean up temporary model file
        if hasattr(self, "model_path") and os.path.exists(self.model_path):
            try:
                os.remove(self.model_path)
                print(f"Deleted temporary model file: {self.model_path}")
            except Exception as e:
                print(f"Warning: could not delete model file: {e}")

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


