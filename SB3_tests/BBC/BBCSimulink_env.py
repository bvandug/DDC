import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matlab.engine
from typing import Optional, Tuple


class BBCSimulinkEnv(gym.Env):
    """
    Simulink-backed Buck-Boost converter env that mirrors the NumPy env's API.

    Observation (float32 [4]): [vC, error, d_error, target]
      vC      : output capacitor voltage (V) (can be negative for inverting)
      error   : vC - target_voltage
      d_error : derivative of error over the last RL step (V/s)
      target  : target_voltage (constant feature per episode unless random_target=True)

    Action (float32 [1]): duty cycle in [0.1, 0.9]

    One RL step = exactly one full PWM period: frame_skip * dt == T_sw.

    Termination/Truncation:
      - After grace_period_steps, terminate early if soft/hard voltage limits or (optionally) inductor current limit is violated.
      - Truncate when time >= max_episode_time.

    Reward (mirrors np_bbc_env.calculate_reward):
      r = exp(-5 * e_norm^2) + 0.5 * progress + band_bonus
          - lam_duty * dduty^2 - lam_i * (|iL|/I_L_MAX)^2  (iL term only if available)

    Notes:
      * This env assumes your Simulink model exposes signals:
          out.voltage  -> vC (scalar timeseries)
          out.tout     -> time vector
        Optionally (if available):
          out.iL       -> inductor current (A)
        If out.iL is not available, the iL regularizer and current-based safety are skipped.

      * The model must have two tunable blocks/params:
          <model>/DutyCycleInput  (scalar value block for duty fraction 0..1)
          <model>/Goal            (scalar value block for target voltage)

      * Ensure the PWM subsystem uses the DutyCycleInput value over the full
        interval [t, t + frame_skip*dt) so the action maps to one PWM period exactly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_name: str = "bbcSim",
        *,
        dt: float = 5e-6,
        frame_skip: int = 10,            # 20 kHz default
        max_episode_time: float = 0.2,   # 0.2 s default (like np env with 4000 steps @ 50 µs)
        grace_period_steps: int = 100,
        target_voltage: float = -30.0,
        random_target: bool = False,
        target_min: float = -49.0,
        target_max: float = -28.0,
        enable_plotting: bool = False,
        use_fast_restart: bool = True,
    ) -> None:
        super().__init__()

        # --- MATLAB engine / model ---
        self.model_name = model_name
        self.dt = float(dt)
        self.frame_skip = int(frame_skip)
        self.T_sw = self.dt * self.frame_skip
        self.max_episode_time = float(max_episode_time)
        self.grace_period_steps = int(grace_period_steps)
        self.random_target = bool(random_target)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.enable_plotting = bool(enable_plotting)
        self.use_fast_restart = bool(use_fast_restart)

        # Safety/scaling (aligned with np env)
        self.I_L_MAX = 20.0  # A
        self._band_e = 0.02  # ±2% band around |target|

        # Reward weights (aligned with np env)
        self._lam_duty = 0.5
        self._lam_i = 0.05
        self._clip_low = -3.0
        self._clip_high = 2.0

        # Target voltage for this episode (set in reset)
        self.target_voltage = float(target_voltage)

        # Action/Observation spaces
        self.action_space = spaces.Box(
            low=np.array([0.1], dtype=np.float32),
            high=np.array([0.9], dtype=np.float32),
            dtype=np.float32,
        )
        high = np.array([np.finfo(np.float32).max] * 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Internal state bookkeeping
        self.time: float = 0.0
        self.current_step: int = 0
        self.prev_error: float = 0.0
        self.prev_duty: float = 0.5
        self.prev_vC: float = 0.0
        self.last_iL: Optional[float] = None

        # --- Start MATLAB + load model ---
        self.eng = matlab.engine.start_matlab()
        self.eng.load_system(self.model_name, nargout=0)
        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

        # Pre-create storage for simple optional plotting (off by default)
        self._times = []
        self._vcs = []
        self._duties = []

    # ====== MATLAB helpers ======
    def _sim_to(self, stop_time: float) -> None:
        """Advance model from current xFinal to the next stop_time."""
        # Toggle FR off so we can provide/load state, then back on for speed
        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "off", nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'LoadInitialState','on', 'InitialState','xFinal',"
            f"'StopTime','{stop_time}', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0,
        )
        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

    def _read_signal(self, name: str) -> Optional[float]:
        """
        Safely read a field from Simulink SimulationOutput 'out' without printing
        MATLAB errors when the field is absent. Returns the last scalar value or None.
        """
        try:
            # does 'out' have this signal?
            has = bool(self.eng.eval(f"any(strcmp(who(out), '{name}'))"))
            if not has:
                return None

            # fetch it now that we know it exists
            val = self.eng.eval(f"out.{name}")
        except Exception:
            return None

        # Accept scalars or numeric arrays/timeseries (take last sample)
        try:
            if isinstance(val, float):
                return float(val)
            # handle numeric arrays returned via MATLAB engine (cell-like)
            return float(val[-1][0])
        except Exception:
            try:
                # sometimes engine returns 1-D arrays
                return float(val[-1])
            except Exception:
                return None


    def _get_vC_t_iL(self) -> Tuple[float, float, Optional[float]]:
        vC = self._read_signal("voltage")
        t = self._read_signal("tout")
        iL = self._read_signal("iL")  # optional
        if vC is None or t is None:
            # Provide clearer failure mode if model outputs are misnamed
            raise RuntimeError(
                "Could not read 'out.voltage' or 'out.tout' from Simulink output.\n"
                "Ensure your model logs these variables as 'voltage' and 'tout'."
            )
        return float(vC), float(t), (None if iL is None else float(iL))

    # ====== Gym API ======
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.time = 0.0
        self.current_step = 0
        self.prev_duty = 0.5

        # Choose/Set target voltage
        if self.random_target:
            # sample uniformly in [target_min, target_max]
            self.target_voltage = float(self.np_random.uniform(low=self.target_min, high=self.target_max))
        # Push target into model
        self.eng.set_param(f"{self.model_name}/Goal", "Value", str(self.target_voltage), nargout=0)

        # (Re)initialize state by running a tiny sim to produce xFinal
        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "off", nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'StopTime','1e-6', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0,
        )
        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

        # Read initial measurement
        vC, t, iL = self._get_vC_t_iL()
        self.time = t
        error = vC - self.target_voltage
        self.prev_error = error
        self.prev_vC = vC
        self.last_iL = iL

        # Clear debug traces
        if self.enable_plotting:
            self._times.clear(); self._vcs.clear(); self._duties.clear()

        obs = np.array([vC, error, 0.0, self.target_voltage], dtype=np.float32)
        info = {
            "iL": (None if iL is None else float(iL)),
            "vC": float(vC),
            "mag_vC": float(abs(vC)),
            "err": float(error),
            "e_norm": float(abs(abs(vC) - abs(self.target_voltage)) / max(abs(self.target_voltage), 1e-3)),
            "dduty": 0.0,
            "in_band": bool(abs(abs(vC) - abs(self.target_voltage)) <= self._band_e * abs(self.target_voltage)),
            "duty_cmd": float(self.prev_duty),
            "eff_duty": float(self.prev_duty),
            "frame_skip": int(self.frame_skip),
            "dt": float(self.dt),
            "T_sw": float(self.T_sw),
        }
        return obs, info

    def step(self, action):
        duty = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        # Apply duty to model for exactly one PWM period
        self.eng.set_param(f"{self.model_name}/DutyCycleInput", "Value", str(duty), nargout=0)
        stop_time = self.time + self.T_sw
        self._sim_to(stop_time)

        # Read outputs
        vC, t, iL = self._get_vC_t_iL()
        self.time = t

        # Observations
        error = vC - self.target_voltage
        d_error = (error - self.prev_error) / self.T_sw
        obs = np.array([vC, error, d_error, self.target_voltage], dtype=np.float32)

        # Reward (mirrors np env)
        reward = self._calculate_reward(duty=duty, vC=vC, iL=iL)

        # Termination / truncation
        terminated = False
        truncated = False
        self.current_step += 1
        if self.current_step > self.grace_period_steps:
            v_abs = abs(vC)
            vref = abs(self.target_voltage)
            v_out_min = 0.1 * vref
            v_out_max = 1.5 * vref
            over_il = (iL is not None) and (abs(iL) > self.I_L_MAX)
            under_v = v_abs < v_out_min
            over_v = v_abs > v_out_max
            if over_il or under_v or over_v:
                reward -= 1000.0
                terminated = True
        if not terminated and self.time >= self.max_episode_time:
            truncated = True

        # Telemetry
        info = {
            "iL": (None if iL is None else float(iL)),
            "vC": float(vC),
            "mag_vC": float(abs(vC)),
            "err": float(error),
            "e_norm": float(abs(abs(vC) - abs(self.target_voltage)) / max(abs(self.target_voltage), 1e-3)),
            "dduty": float(duty - self.prev_duty),
            "in_band": bool(abs(abs(vC) - abs(self.target_voltage)) <= self._band_e * abs(self.target_voltage)),
            "duty_cmd": float(duty),
            "eff_duty": float(duty),
            "frame_skip": int(self.frame_skip),
            "dt": float(self.dt),
            "T_sw": float(self.T_sw),
        }

        # Bookkeeping + optional plotting
        self.prev_error = error
        self.prev_duty = duty
        self.prev_vC = vC
        self.last_iL = iL

        if self.enable_plotting:
            self._times.append(self.time)
            self._vcs.append(vC)
            self._duties.append(duty)

        return obs, float(reward), terminated, truncated, info

    # ====== Reward (mirrors np_bbc_env) ======
    def _calculate_reward(self, duty: float, vC: float, iL: Optional[float]) -> float:
        v_abs = abs(vC)
        vref = abs(self.target_voltage)
        e = abs(v_abs - vref)
        e_norm = e / max(vref, 1e-3)

        prev_e = abs(abs(self.prev_vC) - vref) / max(vref, 1e-3)
        progress = prev_e - e_norm  # >0 if closer this step

        r_track = float(np.exp(-5.0 * (e_norm ** 2)))
        in_band = abs(v_abs - vref) <= self._band_e * vref
        band_bonus = 0.1 if in_band else 0.0

        dduty = duty - self.prev_duty
        r = (
            r_track
            + 0.5 * progress
            + band_bonus
            - self._lam_duty * (dduty ** 2)
        )
        if iL is not None:
            i_norm = abs(iL) / max(self.I_L_MAX, 1e-3)
            r -= self._lam_i * (i_norm ** 2)

        return float(np.clip(r, self._clip_low, self._clip_high))

    # ====== Close ======
    def close(self):
        try:
            if self.use_fast_restart:
                self.eng.set_param(self.model_name, "FastRestart", "off", nargout=0)
            self.eng.close_system(self.model_name, 0, nargout=0)
        except Exception:
            pass
        try:
            self.eng.quit()
        except Exception:
            pass
