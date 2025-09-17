"""Switch-level inverting buck–boost RL environment (JAX-friendly).

Implements a discrete-time, switch-accurate model with ESR, diode
forward drop, optional reverse-recovery, and enforced DCM. One RL
step equals one PWM period (frame_skip × dt). Observations are
[vC, error, d_error, target]; action is a duty ratio in [dmin, dmax].
"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np

class JAXBuckBoostConverterEnv(gym.Env):
    """Inverting buck–boost with MOSFET + diode, DCM enforced.

    State:
        iL (A): Inductor current.
        uC (V): Ideal capacitor voltage (across C).
        vC (V): Output/node voltage (includes ESR drop).

    Capacitor ESR model: vC = uC + R_C · iC, with load R in parallel.

    Observation:
        [vC, error (= vC − target), d_error/dt, target]

    Action:
        duty in [duty_min, duty_max] applied within the PWM period
        using an ON-fraction followed by OFF, without tick quantization.

    Notes:
        - One `step()` integrates exactly one PWM period.
        - Gaussian measurement noise can be added to vC (observation only).
    """
    metadata = {}
    def __init__(
        
        self,
        dt: float = 5e-6,                 # Discrete 5e-06 s
        frame_skip: int = 26,             # 26 -> 7.692 kHz @ 5e-6
        max_episode_steps: int = 4000,
        grace_period_steps: int = 200,

        target_voltage: float = -80.0,

        # Power stage (match Simulink)
        Vin: float = 48.0,
        L: float = 470e-6,
        C: float = 220e-6,
        R_load: float = 20,

        # Parasitics
        Ron_sw: float = 0.1,              # MOSFET on-resistance
        R_L: float = 0,                 # inductor DCR

        
        Vf: float = 0.7,                  # diode forward drop
        Ron_d: float = 0.001,             # diode on-resistance
        R_C: float = 0,                 # capacitor ESR

        # Optional extras
        G_off: float = 0.0,               # leakage to ground (A/V)

        enable_rr: bool = False,          # reverse-recovery (very crude pulse model)
        Qrr: float = 0.0,                 # coulombs
        trr: float = 0.0,                 # seconds

        # PWM quantization
        duty_min: float = 0.1,
        duty_max: float = 0.9,
        pwm_rounding: str = "round",
        voltage_noise_std: float = 0.0, 
    ):
        """Initialize converter parameters, limits, and Gym spaces.

        Args:
            dt (float): Integrator time step (s).
            frame_skip (int): Substeps per PWM period (N).
            max_episode_steps (int): Hard episode length cap (steps).
            grace_period_steps (int): Steps before safety termination applies.
            target_voltage (float): Negative output target (V).
            Vin (float): Input voltage (V).
            L (float): Inductance (H).
            C (float): Capacitance (F).
            R_load (float): Load resistance (Ω).
            Ron_sw (float): MOSFET on-resistance (Ω).
            R_L (float): Inductor DCR (Ω).
            Vf (float): Diode forward drop (V).
            Ron_d (float): Diode on-resistance (Ω).
            R_C (float): Capacitor ESR (Ω).
            G_off (float): Leakage conductance to ground (A/V).
            enable_rr (bool): Enable reverse-recovery pulse model.
            Qrr (float): Diode reverse-recovery charge (C).
            trr (float): Reverse-recovery time constant (s).
            duty_min (float): Minimum duty command.
            duty_max (float): Maximum duty command.
            pwm_rounding (str): Reserved; rounding mode for PWM ticks.
            voltage_noise_std (float): Std. dev. of Gaussian voltage noise.
        """
        super().__init__()

        # --- circuit ---
        self.Vin, self.L, self.C = float(Vin), float(L), float(C)
        self.R = float(R_load)
        self.Ron_sw, self.R_L = float(Ron_sw), float(R_L)
        self.Vf, self.Ron_d = float(Vf), float(Ron_d)
        self.R_C = float(R_C)
        self.G_off = float(G_off)

        # reverse-recovery state
        self.enable_rr, self.Qrr, self.trr = bool(enable_rr), float(Qrr), float(trr)
        self._rr_timer = 0.0
        self._rr_i0 = 0.0
        self._rr_elapsed = 0.0

        # --- timing / limits ---
        self.dt, self.N = float(dt), int(frame_skip)
        self.Tsw = self.dt * self.N
        self.frame_skip = self.N
        self.max_episode_steps = int(max_episode_steps)
        self.grace_period_steps = int(grace_period_steps)

        self.target_voltage = float(target_voltage)
        self.I_L_MAX = 100.0
        self.V_OUT_MAX = abs(self.target_voltage) * 3.0
        self.V_OUT_MIN = abs(self.target_voltage) * 0.05

        # numerical guards (keep obs/reward finite)
        self.V_CLAMP = 1e6
        self.I_CLAMP = 1e3

        # action / obs spaces
        self.duty_min, self.duty_max = float(duty_min), float(duty_max)
        self.pwm_rounding = pwm_rounding.lower()
        self.action_space = spaces.Box(
            low=np.array([self.duty_min], np.float32),
            high=np.array([self.duty_max], np.float32),
            dtype=np.float32,
        )
        high = np.array([np.finfo(np.float32).max]*4, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.voltage_noise_std = float(voltage_noise_std)

        self.reset()

    # ---------- helpers ----------

    def _alpha(self):
        """Return ESR/load scaling factor α = 1 + R_C/R.

        Used to convert between uC (ideal cap voltage) and vC (node voltage)
        under the series ESR model with load resistance R.
        """
        return 1.0 + (self.R_C / max(self.R, 1e-12))

    # ---------- core switch-level integrator (with ESR) ----------

    def _on_step(self, iL, uC, dt):
        """Integrate one ON-substep (diode reverse-biased).

        Assumes switch ON: inductor charges from Vin through Ron and DCR.
        Capacitor discharges through the load and leakage.

        Args:
            iL (float): Inductor current at substep start (A).
            uC (float): Ideal capacitor voltage at substep start (V).
            dt (float): Substep duration (s).

        Returns:
            tuple[float, float, float]: (iL_new, uC_new, vC_new).
        """
        alpha = self._alpha()
        v = uC / alpha

        # Inductor (to Vin via MOSFET Ron + DCR); charges up
        diL = (self.Vin - iL*(self.Ron_sw + self.R_L)) / self.L

        # Capacitor internal voltage discharges into load
        iC = -v / self.R
        # include leakage to ground (leaving node) if any: i_leak = G_off * v
        # KCL already uses only iC here (no node injection), so leakage is just another sink through v
        # We fold leakage into iC equivalently:
        iC -= self.G_off * v

        duC = (iC / self.C)

        iL_new = iL + dt*diL
        uC_new = uC + dt*duC
        v_new  = uC_new / alpha
        return iL_new, uC_new, v_new

    def _off_step(self, iL, uC, dt):
        """Integrate one OFF-substep with DCM handling and optional RR.

        Assumes switch OFF: inductor discharges through diode into node.
        Splits the substep if inductor current crosses zero (DCM), then
        continues with iL = 0 for the remainder. Optionally injects a
        reverse-recovery current pulse when turning off.

        Args:
            iL (float): Inductor current at substep start (A).
            uC (float): Ideal capacitor voltage at substep start (V).
            dt (float): Substep duration (s).

        Returns:
            tuple[float, float, float]: (iL_new, uC_new, vC_new).
        """
        irr = 0.0
        if self.enable_rr and self._rr_timer > 0.0:
            self._rr_elapsed = getattr(self, "_rr_elapsed", 0.0)
            irr = -self._rr_i0 * np.exp(-(self._rr_elapsed / max(self.trr, 1e-12)))
            self._rr_elapsed += dt
            if self._rr_elapsed >= self.trr:
                self._rr_timer = 0.0
                irr = 0.0

        alpha = self._alpha()

        # Algebraic node voltage in OFF with ESR (correct sign!)
        v = (uC - self.R_C * (iL + irr)) / alpha

        # Inductor slope while diode conducts: vL = v - Vf - iL*Rseries
        Rseries = self.Ron_d + self.R_L
        diL = (v - self.Vf - iL*Rseries) / self.L
        iL_pred = iL + dt*diL

        # Will current hit zero within this substep?
        if (diL < 0.0) and (iL > 0.0) and (iL_pred < 0.0):
            # Phase 1: conduct until zero
            dt1 = iL / (-diL)
            iC1  = -(iL + irr) - v/self.R - self.G_off * v
            duC1 = (iC1 / self.C)
            uC1  = uC + dt1*duC1

            # start reverse-recovery at turn-off (optional)
            if self.enable_rr and self.Qrr > 0 and self.trr > 0:
                self._rr_timer = 1.0
                self._rr_elapsed = 0.0
                self._rr_i0 = (self.Qrr / self.trr) * 2.0

            # Phase 2: DCM (iL = 0)
            dt2 = dt - dt1
            if dt2 > 0.0:
                v_mid = uC1 / alpha
                iC2  = -(0.0 + irr) - v_mid/self.R - self.G_off * v_mid
                duC2 = (iC2 / self.C)
                uC2  = uC1 + dt2*duC2
            else:
                uC2 = uC1

            v_new = uC2 / alpha
            return 0.0, uC2, v_new

        # No zero-cross this substep (still conducting)
        iC   = -(iL + irr) - v/self.R - self.G_off * v
        duC  = (iC / self.C)
        uC_new = uC + dt*duC
        iL_new = iL_pred
        if iL_new < 0.0:      # numerical guard
            iL_new = 0.0
            v_new  = uC_new / alpha
        else:
            v_new  = (uC_new - self.R_C * (iL_new + irr)) / alpha
        return iL_new, uC_new, v_new


    # ---------- Gym API ----------

    def reset(self, *, seed=None, options=None):
        """Reset environment to a cold start and return initial observation.

        Zeroes states, clears RR flags and episode stats, and returns the
        noisy initial observation (noise added to vC only).

        Args:
            seed (int | None): RNG seed for Gym API.
            options (dict | None): Reserved for API compliance.

        Returns:
            tuple[np.ndarray, dict]: (obs, info) where obs = [vC, e, de, target].
        """
        super().reset(seed=seed)
        self.time = 0.0
        self.step_count = 0
        self._rr_timer = 0.0
        self._rr_elapsed = 0.0

        #tracking for voltage and duty
        self._ep_duty_sum = 0.0
        self._ep_steps = 0
        self._avg_duty = 0.0
        self._ep_v_sum    = 0.0
        self._avg_voltage = 0.0

        # cold start
        self.iL = 0.0
        self.uC = 0.0
        self.vC = 0.0

        self.prev_vC = self.vC
        self.prev_error = self.vC - getattr(self, "target_voltage", -80.0)
        self.prev_applied_duty = 0.0

        # Add noise to the initial observation as well
        noisy_vC = self.vC + self.np_random.normal(0, self.voltage_noise_std)
        initial_error = noisy_vC - self.target_voltage
        self.prev_error = initial_error # Make sure prev_error is also based on noisy reading
        obs = np.array([noisy_vC, initial_error, 0.0, self.target_voltage], np.float32)
        return obs, {}

    def step(self, action):
        """Advance exactly one PWM period with ON/OFF composition.

        Within the period, applies an ON fraction (k_on + fractional) then
        OFF, integrating substeps sequentially. Observation and error terms
        use the NOISY vC; reward uses true state.

        Args:
            action (array-like): Duty ratio in [duty_min, duty_max].

        Returns:
            tuple:
                - obs (np.ndarray): [vC_noisy, error, d_error, target].
                - reward (float): Scalar reward for this period.
                - terminated (bool): True if safety/goal termination triggered.
                - truncated (bool): True if max_episode_steps reached.
                - info (dict): Diagnostics (iL, vC, dduty, averages, timing).
        """
        # 1) Commanded duty, band-limited
        duty_cmd = float(np.clip(action[0], self.duty_min, self.duty_max))

        # 2) Continuous duty within one period (no tick quantising)
        N = self.N
        x = duty_cmd * N
        k_on = int(np.floor(x + 1e-12))       # full ON substeps
        f = float(x - k_on)                   # fractional ON in [0,1)
        duty_eff = duty_cmd                   # applied duty equals command (unquantised)

        # 3) Integrate exactly one switching period
        iL, uC, vC = self.iL, self.uC, self.vC
        self._rr_timer = 0.0

        # accumulate simple per-slot means (same shape as before)
        vC_sum = 0.0
        iL_sum = 0.0

        # (a) k_on full ON substeps
        for _ in range(k_on):
            iL, uC, vC = self._on_step(iL, uC, self.dt)
            self.time += self.dt
            vC_sum += vC
            iL_sum += iL

        # (b) one mixed slot: ON for dt*f then OFF for dt*(1-f)
        if f > 1e-12:
            # fractional ON
            iL, uC, vC = self._on_step(iL, uC, self.dt * f)
            self.time += self.dt * f
            # fractional OFF
            iL, uC, vC = self._off_step(iL, uC, self.dt * (1.0 - f))
            self.time += self.dt * (1.0 - f)
            # count this mixed slot once in the running mean
            vC_sum += vC
            iL_sum += iL
            # remaining OFF full substeps
            for _ in range(N - k_on - 1):
                iL, uC, vC = self._off_step(iL, uC, self.dt)
                self.time += self.dt
                vC_sum += vC
                iL_sum += iL
        else:
            # no fractional split -> N - k_on full OFF substeps
            for _ in range(N - k_on):
                iL, uC, vC = self._off_step(iL, uC, self.dt)
                self.time += self.dt
                vC_sum += vC
                iL_sum += iL

        # 4) Update & clamp state (unchanged)
        self.iL = float(np.clip(iL, -self.I_CLAMP, self.I_CLAMP))
        self.uC = float(np.clip(uC, -self.V_CLAMP, self.V_CLAMP))
        self.vC = float(np.clip(vC, -self.V_CLAMP, self.V_CLAMP))

        self.step_count += 1

        # --- START: NOISE IMPLEMENTATION ---
        # 1. Get the true voltage and add Gaussian noise to it
        true_vC = self.vC
        noisy_vC = true_vC + self.np_random.normal(0, self.voltage_noise_std)

        # 2. The agent's perception of error is based on the NOISY voltage
        error = noisy_vC - self.target_voltage
        d_error = (error - self.prev_error) / self.Tsw

        # 3. Construct the observation array with the noisy values
        v_obs = float(np.clip(noisy_vC, -self.V_CLAMP, self.V_CLAMP))
        obs = np.array([v_obs, error, d_error, self.target_voltage], np.float32)
        # --- END: NOISE IMPLEMENTATION ---

        # Reward exactly as before (uses duty_eff)
        reward = self._reward(duty_eff)

        # Termination / truncation (unchanged)
        terminated, truncated = False, False
        if self.step_count > self.grace_period_steps:
            if abs(self.iL) > self.I_L_MAX or not ((self.vC) <= self.V_OUT_MAX):
                reward -= 1000.0
                terminated = True
        if not terminated and self.step_count >= self.max_episode_steps:
            truncated = True

        # Period means (same definition as prior code)
        vC_mean_period = vC_sum / float(self.N)
        iL_mean_period = iL_sum / float(self.N)
        on_time = duty_eff * self.frame_skip * self.dt              # exact ON duration this PWM period
        full_on = int(on_time // self.dt)                       # number of full ON substeps
        frac    = (on_time / self.dt) - full_on                 # fractional ON of the next substep in [0,1)

        info = {
            "iL": float(iL),
            "vC": float(vC),
            "mag_vC": abs(float(vC)),
            "err": float(error),
            "e_norm": float(abs(abs(vC) - abs(self.target_voltage)) / max(abs(self.target_voltage), 1e-3)),
            "dduty": float(duty_cmd - self.prev_applied_duty),
            "in_band": bool(abs(abs(vC) - abs(self.target_voltage)) <= 0.2 * abs(self.target_voltage)),
            "duty_cmd": float(duty_cmd),         # commanded duty
            "eff_duty": float(duty_eff),                 # exact (up to FP error)
            "on_steps": int(full_on),                # full ON substeps
            "on_frac": float(frac),                  # fractional part of the boundary substep
            "frame_skip": int(self.frame_skip),
            "dt": float(self.dt),
            "T_sw": float(self.dt * self.frame_skip),
            "avg_duty" : float(self._avg_duty),
            "avg_voltage" : float(self._avg_voltage)

        }

        self.prev_error = error
        self.prev_applied_duty = duty_eff
        self.prev_vC = self.vC
        return obs, float(reward), terminated, truncated, info


    # ---------- reward (linear track, sign-aware, hinge beyond cap) ----------
    def _reward(self, duty):
        """Compute period reward with tracking, progress, and smoothness terms.

        Reward combines:
            - r_track: radial error term (exp of −5·e_norm²) with clamp.
            - progress: change in normalized error since last step.
            - band_bonus: small bonus when e_norm ≤ 0.02.
            - smoothness: quadratic penalty on duty delta, scaled by e_norm.

        Also updates running per-episode averages used for logging.

        Args:
            duty (float): Applied duty for this period.

        Returns:
            float: Clipped reward in [-5.0, 2.0].
        """

        # accumulate per-episode stats (ONE step increment)
        #--------------------------------------------------------------------------
        u = float(np.array(duty).reshape(-1)[0])   # scalar duty in [0,1]
        v = float(self.vC)

        self._ep_duty_sum += u
        self._ep_v_sum    += v
        self._ep_steps    += 1                     # <-- increment ONCE

        self._avg_duty    = self._ep_duty_sum / max(self._ep_steps, 1)
        self._avg_voltage = self._ep_v_sum    / max(self._ep_steps, 1)
        #--------------------------------------------------------------------------
        
        vref = float(self.target_voltage)  # target output voltage
        e_norm = abs(v - vref) / max(abs(vref), 1e-9)  # normalize abs error

        # Gaussian tracking term (decays with squared error)
        exparg = -5.0 * (e_norm ** 2)
        r_track = 0.0 if exparg < -50.0 else float(np.exp(exparg))

        # Progress term: positive if error decreased since last step
        prev_e_norm = abs(float(self.prev_vC) - vref) / max(abs(vref), 1e-9)
        progress = float(np.clip(prev_e_norm - e_norm, -1.0, 1.0))

        # Bonus when inside tight band (≤ 2% error)
        band_bonus = 0.1 if e_norm <= 0.02 else 0.0

        # Smoothness penalty: penalize duty changes, scaled by error
        dduty = 0.0 if self.prev_applied_duty is None else (duty - self.prev_applied_duty)
        dduty_scale = float(np.exp(-6.0 * e_norm))  # more penalty near setpoint

        # Total reward: tracking + progress + bonus − smoothness penalty
        r = r_track + 0.1 * progress + band_bonus - 0.5 * dduty_scale * (dduty ** 2)

        # Clip to avoid extreme values and return
        return float(np.clip(r, -5.0, 2.0))

