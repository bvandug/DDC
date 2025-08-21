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
        quantize_pwm=True, 
        quantize_mode="round"
        
    ):
        super().__init__()

        # In the __init__ method of JAXBuckBoostConverterEnv

        # --- Circuit Parameters ---
        self.Vin       = 48.0           # Input voltage [V]
        self.L         = 220e-6        # Inductance [H] <-- Updated to match Simulink
        self.C         = 100e-6         # Capacitance [F] <-- Updated to match Simulink
        self.R_load    = 20           # Load resistance [Ohm] <-- Updated to match Simulink
        self.Ron_sw    = 0.1            # MOSFET on-resistance [Ohm] <-- Updated to match Simulink
        self.Ron_d     = 0.01         # Diode conduction resistance [Ohm] <-- Updated to match Simulink
        self.Vf        = 0.7            # Diode forward drop [V]

        # --- NEW: Parasitic Components to match Simulink ---
        self.R_L       = 0            # Inductor series resistance (DCR) [Ohm]
        self.Rd_mosfet = 0.001        # MOSFET body diode resistance [Ohm] (though rarely used in this topology)

        # Note: Incorporating capacitor ESR (R_C = 1.0 Ohm) is more complex.
        # It would require changing the state equations because the output voltage
        # would no longer be identical to the capacitor voltage (v_out != vC).
        # For simplicity, we'll focus on the inductor resistance first.

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

        #quantization parameters
        self.quantize_pwm  = bool(quantize_pwm)
        self.quantize_mode = str(quantize_mode)

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
        self.prev_duty    = 0

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
        """One Euler substep of the converter dynamics with parasitics."""
        dt = self.dt if dt_override is None else float(dt_override)
        L  = self.L
        C  = self.C
        R  = self.R_load

        if u == 1:  # Switch is ON
            # ON: inductor sees Vin minus switch + DCR
            diL = (self.Vin - iL * (self.Ron_sw + self.R_L)) / L
            # Cap only supplies the load while ON
            dvC = (-vC / R) / C

            iL_new = iL + dt * diL
            vC_new = vC + dt * dvC
            if self.enforce_dcm and iL_new < 0.0:
                iL_new = 0.0
            return iL_new, vC_new

        # --- OFF ---
        # Include diode + body-diode + DCR series R
        Rseries_off = self.Ron_d + self.Rd_mosfet + self.R_L
        diL_off     = (vC - self.Vf - iL * Rseries_off) / L

        # If iL would hit zero inside this substep, split at the zero crossing.
        if self.enforce_dcm and diL_off < 0.0:
            t_to_zero = iL / (-diL_off) if iL > 0.0 else 0.0
            if 0.0 < t_to_zero <= dt:
                dt1 = float(np.clip(t_to_zero, 0.0, dt))
                # Use average inductor current (iL/2):  dvC = ((+0.5*iL) - vC/R) / C
                dvC_phase1 = ((0.5 * iL) - vC / R) / C
                vC_mid = vC + dt1 * dvC_phase1

                # Remaining time with iL clamped at zero
                dt2 = dt - dt1
                if dt2 > 0.0:
                    dvC_phase2 = (-vC_mid / R) / C
                    vC_new = vC_mid + dt2 * dvC_phase2
                else:
                    vC_new = vC_mid
                return 0.0, vC_new

        # No zero-cross inside this substep
        # Correct KCL: dvC = (iL - vC/R) / C
        dvC    = (iL - vC / R) / C
        iL_new = iL + dt * diL_off
        vC_new = vC + dt * dvC

        # Numerical safety clamp
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
        self.prev_cmd_duty = None
        self.prev_applied_duty = None

        obs = np.array([vC, error, 0.0, self.target_voltage], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        # 1) Commanded duty (clipped to action space)
        duty_cmd = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        N = int(self.frame_skip)

        # 2) Resolve to the duty actually applied this period
        if self.quantize_pwm:
            if self.quantize_mode.lower() == "floor":
                k = int(np.floor(duty_cmd * N + 1e-12))
            else:  # "round" (usually matches Simulink PWM better)
                k = int(np.round(duty_cmd * N))
            kmin = int(np.ceil(self.action_space.low[0]  * N))
            kmax = int(np.floor(self.action_space.high[0] * N))
            k = max(kmin, min(k, kmax))
            duty_eff = k / N
            full_on  = k
            frac     = 0.0
        else:
            duty_eff = duty_cmd
            on_time  = duty_eff * N * self.dt
            full_on  = int(on_time // self.dt)
            frac     = (on_time / self.dt) - full_on

        # 3) Cache prev state
        self.prev_state = self.state.copy()
        prev_error = self.prev_error

        # 4) Integrate exactly one PWM period using the resolved ON schedule
        iL, vC = float(self.state[0]), float(self.state[1])
        for k_step in range(N):
            if k_step < full_on:
                iL, vC = self._integrate_substep(1, iL, vC)
                self.time += self.dt
            elif k_step == full_on and frac > 0.0:
                dt1 = frac * self.dt
                dt2 = self.dt - dt1
                if dt1 > 0.0:
                    iL, vC = self._integrate_substep(1, iL, vC, dt_override=dt1)
                    self.time += dt1
                if dt2 > 0.0:
                    iL, vC = self._integrate_substep(0, iL, vC, dt_override=dt2)
                    self.time += dt2
            else:
                iL, vC = self._integrate_substep(0, iL, vC)
                self.time += self.dt

        # 5) Update state & obs
        self.state = np.array([iL, vC], dtype=float)
        self.current_step += 1
        error   = vC - self.target_voltage
        d_error = (error - prev_error) / (self.dt * self.frame_skip)
        obs = np.array([vC, error, d_error, self.target_voltage], dtype=np.float32)

        # 6) Reward — use what the plant applied (duty_eff)
        reward = self.calculate_reward(duty_eff)

        # 7) Termination / truncation
        terminated = False
        truncated  = False
        if self.current_step > self.grace_period_steps:
            mag_vC = abs(vC)
            if abs(iL) > self.I_L_MAX or (mag_vC < self.V_OUT_MIN) or (mag_vC > self.V_OUT_MAX):
                reward -= 1000.0
                terminated = True
        if not terminated and self.current_step >= self.max_episode_steps:
            truncated = True

        # 8) Clean telemetry
        info = {
            "iL": float(iL),
            "vC": float(vC),
            "mag_vC": abs(float(vC)),
            "err": float(error),
            "e_norm": float(abs(abs(vC) - abs(self.target_voltage)) / max(abs(self.target_voltage), 1e-3)),
            "dduty": float(0.0 if self.prev_applied_duty is None else (duty_eff - self.prev_applied_duty)),
            "in_band": bool(abs(abs(vC) - abs(self.target_voltage)) <= self._band_e * abs(self.target_voltage)),
            "duty_cmd": float(duty_cmd),
            "measured_duty": float(duty_eff),
            "eff_duty": float(duty_eff),
            "on_steps": int(full_on),
            "on_frac":  float(frac),
            "frame_skip": int(self.frame_skip),
            "dt": float(self.dt),
            "T_sw": float(self.dt * self.frame_skip),
        }

        # 9) Bookkeeping
        self.prev_error = error
        self.prev_cmd_duty  = duty_cmd
        self.prev_applied_duty = duty_eff
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

        dduty = 0.0 if self.prev_applied_duty is None else (duty - self.prev_applied_duty)

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
