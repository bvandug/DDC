# ip_simulink_env.py
import numpy as np
import matlab.engine
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


class IPEnv(gym.Env):
    """
    Gymnasium-compatible env that steps a Simulink model one base tick per call.

    Expects MATLAB helpers in the MATLAB path:
      - ip_reset(model, theta0, thetaDot0, dt, angle_block, angvel_block) -> obs(2,)
      - ip_step(model, u, angle_limit) -> (obs(2,), reward, done, t)

    Model wiring assumptions:
      - 'model' is your modified Simulink model (e.g., 'pendulum_core')
      - Torque input is driven from a Data Store Memory named 'u_hold'
      - Two 'To Workspace' blocks exist and are named:
            angle_block   = 'pendulum_core/angle'
            angvel_block  = 'pendulum_core/angle_v'
        (Use their **block paths**, not VariableName fields)
      - The model runs with a fixed-step discrete solver at dt = Ts
      - The model is started and *paused* during reset; each step does 'SimulationCommand','step'
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        model: str = "pendulum_core",
        dt: float = 0.01,
        angle_threshold: float = np.pi / 2,
        max_time: float = 5.0,
        action_low: float = -2.0,
        action_high: float = 2.0,
        angle_block: str = "pendulum_core/angle",
        angvel_block: str = "pendulum_core/angle_v",
        matlab_workdir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.dt = float(dt)
        self.angle_threshold = float(angle_threshold)
        self.max_time = float(max_time)
        self.angle_block = angle_block
        self.angvel_block = angvel_block
        self._rng = np.random.default_rng(seed)
        self._t = 0.0

        # --- Gym spaces ---
        # Action: scalar torque
        self.action_space = spaces.Box(
            low=np.array([action_low], dtype=np.float32),
            high=np.array([action_high], dtype=np.float32),
            dtype=np.float32,
        )
        # Observation: [theta, theta_dot]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        # --- MATLAB Engine startup ---
        self.eng = matlab.engine.start_matlab()
        if matlab_workdir is not None:
            # Ensure MATLAB can see ip_reset.m, ip_step.m, and the model file
            self.eng.addpath(matlab_workdir, nargout=0)
            self.eng.cd(matlab_workdir, nargout=0)

        # Load but do not start/step here; ip_reset handles it
        self.eng.load_system(self.model, nargout=0)

    # Gymnasium API
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        theta0 = float(options.get("theta0", 0.0)) if options else 0.0
        thetaDot0 = float(options.get("thetaDot0", 0.0)) if options else 0.0

        # Start + pause model, prime runtime objects, and read initial obs
        obs_mat = self.eng.ip_reset(
            self.model, theta0, thetaDot0, self.dt,
            self.angle_block, self.angvel_block,
            nargout=1
        )
        obs = np.array(obs_mat, dtype=np.float32).flatten()
        self._t = 0.0
        info = {"time": self._t}
        return obs, info

    def step(self, action):
        # Clip and convert action (handle scalar or array)
        if isinstance(action, (list, tuple, np.ndarray)):
            u = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        else:
            u = float(np.clip(action, self.action_space.low[0], self.action_space.high[0]))

        # One exact fixed-step tick while model is paused
        obs_mat, reward, done_flag, t = self.eng.ip_step(
            self.model, u, float(self.angle_threshold), nargout=4
        )

        obs = np.array(obs_mat, dtype=np.float32).flatten()
        reward = float(reward)
        self._t = float(t)

        terminated = bool(done_flag)                  # angle threshold exceeded
        truncated = bool(self._t >= self.max_time)    # time limit
        info = {"time": self._t}

        return obs, reward, terminated, truncated, info

    def close(self):
    # Stop the Simulink run and shut down MATLAB Engine
        try:
            self.eng.set_param(self.model, 'SimulationCommand', 'stop', nargout=0)
        except Exception:
            pass
        try:
            self.eng.quit()
        except Exception:
            pass