import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matlab.engine
from typing import Optional, Tuple
import os, shutil, tempfile, uuid


class DiscretizeDutyWrapper(gym.ActionWrapper):
    def __init__(self, env, n_bins: int = 51, low: float = 0.0, high: float = 1.0, avoid_edges: bool = False):
        """ Create a discrete-to-continuous duty mapping wrapper.

            Maps `Discrete(n_bins)` actions to a scalar duty in `[low, high]`,
            optionally nudging away from exact edges to avoid plant issues. The
            underlying environment must expose a Box action space (shape (1,)).

            Parameters
            ----------
            env :
                Base environment with a continuous action space.
            n_bins : int, optional
                Number of discrete duty bins. Default 51.
            low : float, optional
                Minimum duty value. Default 0.0.
            high : float, optional
                Maximum duty value. Default 1.0.
            avoid_edges : bool, optional
                If True, shrink the effective range slightly so the first/last bins
                are inside (low, high). Default False.
        """

        super().__init__(env)
        assert hasattr(env.action_space, "low") and hasattr(env.action_space, "high"), \
            "Underlying env must have a Box action space."
        self.n_bins = int(n_bins)
        self.low = float(low)
        self.high = float(high)
        self.avoid_edges = bool(avoid_edges)

        if self.avoid_edges and self.n_bins > 1:
            eps = 0.5 / (self.n_bins - 1)
            lo = self.low + eps * (self.high - self.low)
            hi = self.high - eps * (self.high - self.low)
            self._bins = np.linspace(lo, hi, self.n_bins, dtype=np.float32)
        else:
            self._bins = np.linspace(self.low, self.high, self.n_bins, dtype=np.float32)

        self.action_space = spaces.Discrete(self.n_bins)

    def action(self, act_idx: int):
        """ Convert a discrete index to a (1,) float32 duty command.

            Parameters
            ----------
            act_idx : int
                Discrete action index in `[0, n_bins-1]` (clipped if out of range).

            Returns
            -------
            np.ndarray
                Array of shape (1,) containing the mapped duty value.
        """

        idx = int(np.clip(act_idx, 0, self.n_bins - 1))
        return np.array([self._bins[idx]], dtype=np.float32)

class BBCSimulinkEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_name: str = "bbcSim",
        *,
        dt: float = 5e-6,
        frame_skip: int = 26,            
        max_episode_time: float = 0.52,
        grace_period_steps: int = 100,
        target_voltage: float = -80.0,
        random_target: bool = False,
        target_min: float = -49.0,
        target_max: float = -28.0,
        enable_plotting: bool = False,
        use_fast_restart: bool = True,
        quantize_pwm: bool = False,
        quantize_mode: str = 'round',
        voltage_noise_std: float = 0.0,
    ) -> None:
        """Initialize a Simulink-backed buck–boost Gym environment.

            Starts a MATLAB engine, creates a **unique** copy of the `.slx` model in
            a temp directory (to isolate instances), configures timing (dt, frame
            skip → `T_sw`), reward/scaling, safety thresholds, observation/action
            spaces, and optional PWM quantization and plotting buffers.

            Parameters
            ----------
            model_name : str, optional
                Base Simulink model name (without the unique suffix). Default "bbcSim".
            dt : float, optional
                Fixed-step solver timestep (seconds). Default 5e-6.
            frame_skip : int, optional
                Number of base-steps per RL step (PWM period). Default 26.
            max_episode_time : float, optional
                Time limit (seconds) for truncation. Default 0.52.
            grace_period_steps : int, optional
                Steps before safety terminations are enforced. Default 100.
            target_voltage : float, optional
                Initial target voltage (volts, negative for inverting). Default -80.0.
            random_target : bool, optional
                If True, sample target uniformly in [target_min, target_max] at reset.
            target_min : float, optional
                Minimum random target (V). Default -49.0.
            target_max : float, optional
                Maximum random target (V). Default -28.0.
            enable_plotting : bool, optional
                If True, accumulate simple time/voltage/duty traces. Default False.
            use_fast_restart : bool, optional
                Enable Simulink Fast Restart. Default True.
            quantize_pwm : bool, optional
                Quantize duty to `on_steps / frame_skip`. Default False.
            quantize_mode : {'round','floor'}, optional
                Rounding mode for PWM quantization. Default 'round'.
            voltage_noise_std : float, optional
                Std-dev of zero-mean Gaussian noise added to **observed** vC. Default 0.0.
        """

        super().__init__()

        self.model_name = model_name
        self.dt = float(dt)
        self.frame_skip = int(frame_skip)
        self.T_sw = self.dt * self.frame_skip
        self.quantize_pwm = bool(quantize_pwm)
        self.quantize_mode = str(quantize_mode)
        self.voltage_noise_std = float(voltage_noise_std)
        self.prev_cmd_duty = None
        self.prev_applied_duty = None
        self.max_episode_time = float(max_episode_time)
        self.grace_period_steps = int(grace_period_steps)
        self.random_target = bool(random_target)
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.enable_plotting = bool(enable_plotting)
        self.use_fast_restart = bool(use_fast_restart)

        self.I_L_MAX = 20.0 
        self._band_e = 0.02  # ±2% band around |target|

        # Reward weights
        self._lam_duty = 0.5
        self._lam_i = 0.05
        self._clip_low = -3.0
        self._clip_high = 2.0

        # Target voltage for this episode
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

        self.eng = matlab.engine.start_matlab()

        self.base_model_name = self.model_name          
        unique_id = uuid.uuid4().hex[:8]
        self.model_name = f"{self.base_model_name}_{unique_id}"

        self.model_path = os.path.join(tempfile.gettempdir(), f"{self.model_name}.slx")
        shutil.copy(f"{self.base_model_name}.slx", self.model_path)

        self.eng.load_system(self.model_path, nargout=0)
        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)


        self._times = []
        self._vcs = []
        self._duties = []

    def _sim_to(self, stop_time: float) -> None:
        """ Advance the Simulink model from current `xFinal` to `stop_time`.

            Temporarily disables Fast Restart to load/provide state, runs the model
            with fixed-step solver and `FixedStep=dt`, saves the final state back to
            `xFinal`, and re-enables Fast Restart if configured.

            Parameters
            ----------
            stop_time : float
                Absolute simulation time (seconds) to stop at.
        """
        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "off", nargout=0)
        self.eng.set_param(self.model_name, 'SolverType', 'Fixed-step', nargout=0)
        self.eng.set_param(self.model_name, 'FixedStep', str(self.dt), nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'LoadInitialState','on', 'InitialState','xFinal',"
            f"'StopTime','{stop_time}', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0,
        )
        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

    def _read_signal(self, name: str) -> Optional[float]:
        """ Safely fetch a scalar signal from Simulink `out` without noisy errors.

            Checks whether `out` contains the requested field before reading it.
            Accepts scalars or arrays/timeseries-like outputs and returns the last
            sample when applicable.

            Parameters
            ----------
            name : str
                Field name inside the Simulink `out` object (e.g., "voltage", "tout").

            Returns
            -------
            float | None
                The last scalar value if present and readable; otherwise None.
        """

        try:
            has = bool(self.eng.eval(f"any(strcmp(who(out), '{name}'))"))
            if not has:
                return None

            val = self.eng.eval(f"out.{name}")
        except Exception:
            return None

        try:
            if isinstance(val, float):
                return float(val)
            return float(val[-1][0])
        except Exception:
            try:
                return float(val[-1])
            except Exception:
                return None


    def _get_vC_t_iL(self) -> Tuple[float, float, Optional[float]]:
        """Read output voltage, time, and optional inductor current from `out`.

            Returns
            -------
            tuple[float, float, float | None]
                `(vC, t, iL)` where `iL` may be None if not logged.

            Raises
            ------
            RuntimeError
                If either `out.voltage` or `out.tout` is missing/unreadable.
        """

        vC = self._read_signal("voltage")
        t = self._read_signal("tout")
        iL = self._read_signal("iL")
        if vC is None or t is None:
            raise RuntimeError(
                "Could not read 'out.voltage' or 'out.tout' from Simulink output.\n"
                "Ensure your model logs these variables as 'voltage' and 'tout'."
            )
        return float(vC), float(t), (None if iL is None else float(iL))

    # Gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """ Reset the environment and return the initial observation/info.

            Sets time and counters to zero, optionally samples a random target,
            pushes target and duty=0 into the model, runs a tiny sim to seed
            `xFinal`, reads initial outputs, computes initial error, clears optional
            plot buffers, and returns the observation and telemetry info.

            Parameters
            ----------
            seed : int | None, optional
                Per-episode RNG seed forwarded to Gym's seeding utilities.
            options : dict | None, optional
                Unused placeholder for Gymnasium compatibility.

            Returns
            -------
            tuple[np.ndarray, dict]
                `obs` = `[vC, error, d_error=0, target]` (float32),
                `info` includes keys like `iL`, `vC`, `mag_vC`, `e_norm`, `dduty`,
                `in_band`, `duty_cmd`, `eff_duty`, and timing fields.
        """

        super().reset(seed=seed)
        self.time = 0.0
        self.current_step = 0
        self.prev_cmd_duty = 0
        self.prev_applied_duty = 0

        if self.random_target:
            self.target_voltage = float(self.np_random.uniform(low=self.target_min, high=self.target_max))
        self.eng.set_param(f"{self.model_name}/Goal", "Value", str(self.target_voltage), nargout=0)
        self.eng.set_param(f"{self.model_name}/DutyCycleInput", "Value", '0.0', nargout=0)

        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "off", nargout=0)
        self.eng.set_param(self.model_name, 'SolverType', 'Fixed-step', nargout=0)
        self.eng.set_param(self.model_name, 'FixedStep', str(self.dt), nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}', 'StopTime','0', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0,
        )
        if self.use_fast_restart:
            self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

        vC, t, iL = self._get_vC_t_iL()
        self.time = t
        noisy_vC = vC + self.np_random.normal(0.0, self.voltage_noise_std)
        error = noisy_vC - self.target_voltage
        self.prev_error = error
        self.prev_vC = vC
        self.last_iL = iL

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
            "duty_cmd": 0.0,
            "eff_duty": 0.0,
            "measured_duty": 0.0,
            "frame_skip": int(self.frame_skip),
            "dt": float(self.dt),
            "T_sw": float(self.T_sw),
        }
        print("Sim reset -> t=%.9f, vC=%.3f V" % (self.time, self.prev_vC))
        return obs, info

    def step(self, action):
        """Advance exactly one PWM period using the provided duty command.

            Applies the (optionally quantized) duty for one period (`T_sw`), runs
            the Simulink model, reads outputs, forms the observation, computes the
            reward, evaluates safety terminations after a grace period, handles
            time-limit truncation, and returns the standard Gym 5-tuple.

            Parameters
            ----------
            action : array_like
                Duty command in `[0.1, 0.9]` (scalar or (1,) array).

            Returns
            -------
            tuple
                `(obs, reward, terminated, truncated, info)` where:
                - `obs` is `[vC, error, d_error, target]` (float32),
                - `reward` is a clipped scalar,
                - `terminated` reflects safety violations after grace,
                - `truncated` is True when `time >= max_episode_time`,
                - `info` contains telemetry (iL, duties, timing, etc.).
        """

        duty_cmd = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        if self.quantize_pwm:
            N = int(self.frame_skip)
            if self.quantize_mode.lower().startswith('f'):
                on_steps = int(np.floor(duty_cmd * N))
            else:
                on_steps = int(np.round(duty_cmd * N))
            on_steps = int(np.clip(on_steps, 0, N))
            eff_duty = on_steps / float(N)
        else:
            eff_duty = duty_cmd
            N = int(self.frame_skip)
            on_steps = int(np.round(eff_duty * N))
        # Apply 'eff_duty' for exactly one PWM period
        self.eng.set_param(f"{self.model_name}/DutyCycleInput", "Value", str(eff_duty), nargout=0)
        stop_time = self.time + self.T_sw
        self._sim_to(stop_time)

        vC, t, iL = self._get_vC_t_iL()
        self.time = t

        # Observations
        noisy_vC = vC + self.np_random.normal(0.0, self.voltage_noise_std)
        error = noisy_vC - self.target_voltage
        d_error = (error - self.prev_error) / self.T_sw
        obs = np.array([vC, error, d_error, self.target_voltage], dtype=np.float32)

        reward = self._calculate_reward(duty=eff_duty, vC=vC, iL=iL)

        # Termination
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

        info = {
            "iL": (None if iL is None else float(iL)),
            "vC": float(vC),
            "mag_vC": float(abs(vC)),
            "err": float(error),
            "e_norm": float(abs(abs(vC) - abs(self.target_voltage)) / max(abs(self.target_voltage), 1e-3)),
            "dduty": float(0.0 if self.prev_applied_duty is None else (eff_duty - self.prev_applied_duty)),
            "in_band": bool(abs(abs(vC) - abs(self.target_voltage)) <= self._band_e * abs(self.target_voltage)),
            "duty_cmd": float(duty_cmd),
            "eff_duty": float(eff_duty),
            "measured_duty": float(eff_duty),
            "on_steps": int(on_steps),
            "on_frac": float(eff_duty * self.frame_skip - on_steps),
            "frame_skip": int(self.frame_skip),
            "dt": float(self.dt),
            "T_sw": float(self.T_sw),
        }

        self.prev_error = error
        self.prev_vC = vC
        self.last_iL = iL
        self.prev_cmd_duty = duty_cmd
        self.prev_applied_duty = eff_duty

        if self.enable_plotting:
            self._times.append(self.time)
            self._vcs.append(vC)
            self._duties.append(eff_duty)

        return obs, float(reward), terminated, truncated, info

    def _calculate_reward(self, duty: float, vC: float, iL: Optional[float]) -> float:
        """ Compute the scalar reward (NumPy-parity, sign-aware formulation).

            Reward components:
            - Tracking: `exp(-5 * e_norm^2)` with `e_norm = |vC - vref| / |vref|`
            - Progress: `+ 0.1 * clip(prev_e_norm - e_norm, -1, 1)`
            - Band bonus: `+0.1` if `e_norm <= 0.02`
            - Smoothness penalty: `-0.5 * exp(-6 * e_norm) * (Δduty)^2`
            The result is clipped to [-5.0, 2.0].

            Parameters
            ----------
            duty : float
                Effective duty applied this step (after quantization).
            vC : float
                Measured output capacitor voltage (V).
            iL : float | None
                Inductor current (unused here but kept for parity/extensibility).

            Returns
            -------
            float
                Reward value after clipping.
        """

        vref = float(self.target_voltage)
        e_norm = abs(vC - vref) / max(abs(vref), 1e-9)

        exparg = -5.0 * (e_norm ** 2)
        r_track = 0.0 if exparg < -50.0 else float(np.exp(exparg))

        prev_e_norm = abs(self.prev_vC - vref) / max(abs(vref), 1e-9)
        progress = float(np.clip(prev_e_norm - e_norm, -1.0, 1.0))

        band_bonus = 0.1 if e_norm <= 0.02 else 0.0

        dduty = 0.0 if self.prev_applied_duty is None else (duty - self.prev_applied_duty)
        dduty_scale = float(np.exp(-6.0 * e_norm))

        r = r_track + 0.1 * progress + band_bonus - 0.5 * dduty_scale * (dduty ** 2)

        return float(np.clip(r, -5.0, 2.0))


    def close(self):
        """ Shut down MATLAB and clean up the unique model copy and caches.

            Attempts to disable Fast Restart, close the loaded system, quit the
            MATLAB engine, delete the temporary `.slx` copy, remove `slprj/<model>`
            artifacts, and clean associated autosave/`.slxc` files. Errors are
            suppressed to make teardown robust.
        """

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

        try:
            if hasattr(self, "model_path") and os.path.exists(self.model_path):
                os.remove(self.model_path)
        except Exception:
            pass

        try:
            slprj_model_dir = os.path.join(os.getcwd(), "slprj", self.model_name)
            if os.path.exists(slprj_model_dir):
                shutil.rmtree(slprj_model_dir)
        except Exception:
            pass

        try:
            base = os.path.splitext(self.model_name)[0]
            for fname in (f"{base}.slx.autosave", f"{base}.slxc"):
                fpath = os.path.join(os.getcwd(), fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
        except Exception:
            pass
