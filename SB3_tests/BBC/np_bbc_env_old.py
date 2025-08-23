import gymnasium as gym
from gymnasium import spaces
import numpy as np

class JAXBuckBoostConverterEnv(gym.Env):
    """
    Discrete-time inverting buck-boost converter environment with PWM resolved
    over a full switching period per RL step (frame_skip * dt == T_sw).

    Observation: [vC, error, d_error, target]
      vC      : output capacitor voltage (can be negative for inverting)
      error   : vC - target_voltage
      d_error : derivative of error over the previous RL step
      target  : target_voltage (constant feature)

    Action: duty cycle in [0.1, 0.9]
    Reward: smooth, monotonic tracking with band bonus, progress term,
            and small penalties on duty slew and inductor current magnitude.

    Termination:
      - After grace period, terminate on |iL| > I_L_MAX or
        |vC| < V_OUT_MIN or |vC| > V_OUT_MAX, with a large penalty.
      - Truncate on max_episode_steps.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        dt: float = 5e-6,
        max_episode_steps: int = 4000,
        frame_skip: int = 20,
        grace_period_steps: int = 100,
        target_voltage: float = -30.0,
        enforce_dcm: bool = True,          # NEW: prevent negative inductor current (DCM)
        
    ):
        super().__init__()

        # --- Circuit Parameters ---
        self.Vin       = 48.0           # Input voltage [V]
        self.L         = 470e-6          # Inductance [H]
        self.C         = 220e-6         # Capacitance [F]
        self.R_load    = 20.0           # Load resistance [Ohm] (seen by vC)
        self.Ron_sw    = 0.05           # MOSFET on-resistance [Ohm]
        self.Ron_d     = 0.05           # Diode conduction resistance [Ohm]
        self.Vf        = 0.7            # Diode forward drop [V]

        # Simulation parameters
        self.dt                 = dt
        self.frame_skip         = frame_skip
        self.grace_period_steps = grace_period_steps
        self.max_episode_steps  = max_episode_steps
        self.target_voltage     = float(target_voltage)
        self.enforce_dcm        = bool(enforce_dcm)

        # Safety limits (scale with |target|)
        self.I_L_MAX   = 20.0  # [A]
        self.V_OUT_MAX = abs(self.target_voltage) * 1.5
        self.V_OUT_MIN = abs(self.target_voltage) * 0.1

        self.exact_duty = True

        # Action: duty cycle
        self.action_space = spaces.Box(
            low=np.array([0.1], dtype=np.float32),
            high=np.array([0.9], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: [vC, error, d_error, target]
        high = np.array([np.finfo(np.float32).max]*4, dtype=np.float32)
        low  = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- State ---
        self.time         = 0.0
        self.state        = np.zeros(2, dtype=float)   # [iL, vC]
        self.prev_state   = self.state.copy()
        self.current_step = 0
        self.prev_error   = 0.0
        self.prev_duty    = 0.5

        # Reward configuration (tunable)
        self._lam_duty  = 0.5
        self._lam_i     = 0.05
        self._band_e    = 0.02  # ±2% band (tweak to 0.03–0.05 if you prefer wider)
        self._clip_low  = -3.0  # widen if using obs-only normalization
        self._clip_high =  2.0

    # ---------- Helpers ----------

    def _integrate_substep(
        self, u: int, iL: float, vC: float, dt_override: float | None = None
    ) -> tuple[float, float]:
        """One Euler substep of the converter dynamics. If dt_override is given, use it."""
        dt = self.dt if dt_override is None else float(dt_override)
        L  = self.L
        C  = self.C
        R  = self.R_load

        if u == 1:
            diL = (self.Vin - self.Ron_sw * iL) / L
            dvC = (-vC / R) / C
            iL_new = iL + dt * diL
            vC_new = vC + dt * dvC
            if self.enforce_dcm and iL_new < 0.0:
                iL_new = 0.0
            return iL_new, vC_new

        diL_off = (vC - self.Vf - self.Ron_d * iL) / L

        if self.enforce_dcm and diL_off < 0.0:
            if iL + dt * diL_off <= 0.0 and iL > 0.0:
                dt1 = iL / (-diL_off)
                dt1 = float(np.clip(dt1, 0.0, dt))
                dvC_on = (-iL - vC / R) / C           # diode ON portion
                vC_mid = vC + dt1 * dvC_on
                dt2 = dt - dt1
                if dt2 > 0.0:
                    dvC_off = (-vC_mid / R) / C      # no inductor once iL hits 0
                    vC_new  = vC_mid + dt2 * dvC_off
                else:
                    vC_new  = vC_mid
                return 0.0, vC_new

        dvC    = (-iL - vC / R) / C                   # regular OFF update
        iL_new = iL + dt * diL_off
        vC_new = vC + dt * dvC

        if self.enforce_dcm and iL_new < 0.0:
            iL_new = 0.0
            vC_new = vC + dt * ((-vC / R) / C)
        return iL_new, vC_new


    # ---------- Gym API ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.time         = 0.0
        self.current_step = 0

        # Start near zero state
        iL = 0.0
        vC = 0.0
        self.state      = np.array([iL, vC], dtype=float)
        self.prev_state = self.state.copy()

        error           = vC - self.target_voltage
        self.prev_error = error
        self.prev_duty  = 0.5

        obs = np.array([vC, error, 0.0, self.target_voltage], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        duty = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        # Cache previous state for reward progress term
        self.prev_state = self.state.copy()
        prev_error = self.prev_error

        # --- Exact-duty PWM over one full switching period ---
        iL, vC = float(self.state[0]), float(self.state[1])

        on_time = duty * self.frame_skip * self.dt              # exact ON duration this PWM period
        full_on = int(on_time // self.dt)                       # number of full ON substeps
        frac    = (on_time / self.dt) - full_on                 # fractional ON of the next substep in [0,1)

        for k in range(self.frame_skip):
            if k < full_on:
                # full ON substeps
                iL, vC = self._integrate_substep(1, iL, vC)     # dt = self.dt
                self.time += self.dt
            elif k == full_on and frac > 0.0:
                # split this substep: ON for dt1, then OFF for dt2, so total ON-time matches 'duty'
                dt1 = frac * self.dt
                dt2 = self.dt - dt1
                if dt1 > 0.0:
                    iL, vC = self._integrate_substep(1, iL, vC, dt_override=dt1)
                    self.time += dt1
                if dt2 > 0.0:
                    iL, vC = self._integrate_substep(0, iL, vC, dt_override=dt2)
                    self.time += dt2
            else:
                # remaining substeps OFF
                iL, vC = self._integrate_substep(0, iL, vC)
                self.time += self.dt

        # Update state
        self.state = np.array([iL, vC], dtype=float)
        self.current_step += 1

        # Observations
        error   = vC - self.target_voltage
        d_error = (error - prev_error) / (self.dt * self.frame_skip)
        obs = np.array([vC, error, d_error, self.target_voltage], dtype=np.float32)

        # Reward
        reward = self.calculate_reward(duty)

        # --- termination / truncation ---
        terminated = False
        truncated  = False
        if self.current_step > self.grace_period_steps:
            mag_vC = abs(vC)
            over_il = abs(iL) > self.I_L_MAX
            under_v = mag_vC < self.V_OUT_MIN
            over_v  = mag_vC > self.V_OUT_MAX
            if over_il or under_v or over_v:
                reward -= 1000.0
                terminated = True
        if not terminated and self.current_step >= self.max_episode_steps:
            truncated = True

        # Telemetry (now includes exact duty info)
        info = {
            "iL": float(iL),
            "vC": float(vC),
            "mag_vC": abs(float(vC)),
            "err": float(error),
            "e_norm": float(abs(abs(vC) - abs(self.target_voltage)) / max(abs(self.target_voltage), 1e-3)),
            "dduty": float(duty - self.prev_duty),
            "in_band": bool(abs(abs(vC) - abs(self.target_voltage)) <= self._band_e * abs(self.target_voltage)),
            "duty_cmd": float(duty),
            "eff_duty": float(duty),                 # exact (up to FP error)
            "on_steps": int(full_on),                # full ON substeps
            "on_frac": float(frac),                  # fractional part of the boundary substep
            "frame_skip": int(self.frame_skip),
            "dt": float(self.dt),
            "T_sw": float(self.dt * self.frame_skip),
        }

        # Bookkeeping
        self.prev_error = error
        self.prev_duty  = duty

        return obs, float(reward), terminated, truncated, info


    # ---------- Reward ----------

    def calculate_reward(self, duty: float) -> float:
        """
        Monotonic tracking with progress, duty slew, and iL regularization.
        Uses magnitudes so the same shaping works for negative targets.
        """
        v_abs = abs(float(self.state[1]))
        vref  = abs(self.target_voltage)
        i_abs = abs(float(self.state[0]))

        e       = abs(v_abs - vref)
        e_norm  = e / max(vref, 1e-3)

        prev_v_abs  = abs(float(self.prev_state[1]))
        prev_e_norm = abs(prev_v_abs - vref) / max(vref, 1e-3)
        progress    = prev_e_norm - e_norm  # positive if improved this step

        dduty = duty - self.prev_duty

        # Smooth, bounded tracking term (Gaussian-like)
        r_track = float(np.exp(-5.0 * (e_norm ** 2)))

        # Band bonus keeps inside-band preferable
        in_band = (abs(v_abs - vref) <= self._band_e * vref)
        band_bonus = 0.1 if in_band else 0.0

        # Regularizers
        i_norm = i_abs / max(self.I_L_MAX, 1e-3)
        r = (
            r_track
            + 0.5 * progress
            + band_bonus
            - self._lam_duty * (dduty ** 2)
            - self._lam_i    * (i_norm ** 2)
        )

        # If you're using reward normalization in VecNormalize(reward=True),
        # consider returning r without clipping to avoid double clipping.
        r = float(np.clip(r, self._clip_low, self._clip_high))
        return r

    # ---------- Render / Close ----------

    def render(self, mode: str = "human"):
        return None

    def close(self):
        return None
