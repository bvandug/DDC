import gym
from gym import spaces
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental.ode import odeint
import numpy as np

class JAXBuckBoostConverterEnv(gym.Env):
    def __init__(self, dt=5e-6, max_episode_time=0.03, target_voltage=-30.0):
        super().__init__()

        # --- Updated Circuit Parameters from Simulink ---
        self.Vin = 48.0
        self.L = 220e-6         # H
        self.C = 100e-6         # F
        self.R = 5.1            # Ohms
        self.Ron_switch = 0.1   # Ohms (MOSFET)
        self.Ron_diode = 0.01   # Ohms (Diode)
        self.Vf_diode = 0.0     # V (Diode forward voltage)

        self.fsw = 10e3
        self.Tsw = 1 / self.fsw

        self.dt = dt
        self.max_episode_time = max_episode_time
        self.target_voltage = target_voltage

        # --- Initial State: [iL, vC] ---
        self.time = 0.0
        self.state = jnp.array([0.0, 0.0])
        self.prev_state = jnp.array([0.0, 0.0])

        # --- Safety Limits ---
        self.I_L_MAX = 20.0  # Maximum inductor current (A)
        self.V_OUT_MAX = abs(target_voltage) * 1.5  # 150% of target
        self.V_OUT_MIN = abs(target_voltage) * 0.1   # 10% of target

        # --- Reward weights (adjusted for better balance) ---
        self.w_voltage = 1.0     # Voltage tracking (reduced to prevent domination)
        self.w_efficiency = 0.5  # Efficiency
        self.w_stability = 0.5   # Stability
        self.w_constraint = 2.0  # Constraint penalties (increased for safety)

        # --- Gym spaces ---
        self.action_space = spaces.Box(low=0.1, high=0.9, shape=(1,), dtype=np.float32)
        high = np.finfo(np.float32).max
        self.observation_space = spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float32)

        self.prev_error = 0.0

    @staticmethod
    @jit
    def pwm(t, D, Tsw):
        return jnp.where((t % Tsw) < D * Tsw, 1.0, 0.0)

    def buck_boost_dynamics(self, x, t, D):
        iL, vC = x
        u = self.pwm(t, D, self.Tsw)

        # When switch is ON: inductor charges from Vin, output isolated
        def on_state(_):
            diL = (self.Vin - iL * self.Ron_switch) / self.L
            dvC = -vC / (self.R * self.C)
            return jnp.array([diL, dvC])

        # When switch is OFF: inductor discharges through diode to output
        def off_state(_):
            diL = (-vC - iL * self.Ron_diode - self.Vf_diode) / self.L
            dvC = (iL - vC / self.R) / self.C
            return jnp.array([diL, dvC])

        return lax.cond(u == 1.0, on_state, off_state, operand=None)

    def voltage_tracking_reward(self, v_out, v_ref):
        """Primary reward: scaled tracking error with reasonable bounds"""
        error = abs(v_out - v_ref)
        relative_error = error / abs(v_ref)
        # Use bounded exponential to prevent extreme negative values
        bounded_penalty = min(relative_error, 5.0)  # Cap at 5x target voltage error
        return -bounded_penalty

    def efficiency_reward(self, state, duty):
        """Reward based on power conversion efficiency"""
        iL, vC = state
        
        # Output power (always positive)
        P_out = abs(vC * vC / self.R)
        
        # Input power estimation (always positive)
        P_in = abs(self.Vin * iL * duty) + 1e-6  # Small epsilon to avoid division by zero
        
        if P_in > 1e-3 and P_out > 1e-3:  # Only calculate if meaningful power levels
            efficiency = min(P_out / P_in, 1.0)  # Cap at 100%
            return 2.0 * (efficiency - 0.5)  # Scale to [-1, 1] range
        else:
            return 0.0  # Neutral reward for very low power

    def stability_reward(self, prev_state, current_state):
        """Penalize rapid changes in state variables"""
        dv_dt = abs(current_state[1] - prev_state[1]) / self.dt
        di_dt = abs(current_state[0] - prev_state[0]) / self.dt
        
        # Normalize by typical values
        voltage_change_penalty = dv_dt / abs(self.target_voltage)
        current_change_penalty = di_dt / 10.0  # Normalize by 10A
        
        return -0.1 * (voltage_change_penalty + current_change_penalty)

    def constraint_violations(self, state, action):
        """Heavy penalties for constraint violations"""
        iL, vC = state
        duty = action[0]
        penalty = 0.0
        
        # Current limit violation
        if abs(iL) > self.I_L_MAX:
            penalty += 10.0 * (abs(iL) - self.I_L_MAX)**2
        
        # Voltage limit violations
        v_out_abs = abs(vC)
        if v_out_abs > self.V_OUT_MAX:
            penalty += 20.0 * ((v_out_abs - self.V_OUT_MAX) / self.V_OUT_MAX)**2
        elif v_out_abs < self.V_OUT_MIN and self.time > 0.005:  # Allow startup transient
            penalty += 5.0
        
        # Duty cycle should be within bounds (already clipped, but double-check)
        if duty < 0.1 or duty > 0.9:
            penalty += 50.0
        
        return penalty

    def calculate_reward(self, prev_state, current_state, action):
        """Multi-objective reward function with improved scaling"""
        iL, vC = current_state
        
        # Individual reward components
        voltage_reward = self.voltage_tracking_reward(vC, self.target_voltage)
        efficiency_reward = self.efficiency_reward(current_state, action[0])
        stability_reward = self.stability_reward(prev_state, current_state)
        constraint_penalty = self.constraint_violations(current_state, action)
        
        # Weighted combination with improved scaling
        total_reward = (self.w_voltage * voltage_reward + 
                       self.w_efficiency * efficiency_reward + 
                       self.w_stability * stability_reward - 
                       self.w_constraint * constraint_penalty)
        
        # Additional bonuses for good performance
        error = abs(vC - self.target_voltage)
        relative_error = error / abs(self.target_voltage)
        
        # Bonus for tight regulation (within 5% of target) - more achievable
        if relative_error < 0.05 and self.time > 0.01:  # After initial transient
            total_reward += 5.0
        
        # Bonus for reasonable regulation (within 10% of target)
        elif relative_error < 0.10 and self.time > 0.01:
            total_reward += 2.0
        
        # Bonus for reaching reasonable steady state quickly
        if relative_error < 0.15 and self.time < 0.005:
            total_reward += 1.0
        
        # Debug: Add minimum reward to prevent all zeros
        total_reward = max(total_reward, -50.0)  # Prevent extreme negative rewards
        
        return total_reward

    def step(self, action):
        duty = float(np.clip(action[0], 0.1, 0.9))
        
        # Store previous state for stability calculation
        self.prev_state = self.state.copy()
        
        t_span = jnp.linspace(self.time, self.time + self.dt, 2)
        self.state = odeint(self.buck_boost_dynamics, self.state, t_span, duty)[-1]

        self.time += self.dt
        vC = float(self.state[1])
        error = vC - self.target_voltage
        d_error = (error - self.prev_error) / self.dt

        # Calculate comprehensive reward
        #reward = self.calculate_reward(self.prev_state, self.state, action)
        reward = -(error**2)
        print(f"[step] t={self.time:.4f}, iL={float(self.state[0]):.2f}, vC={float(self.state[1]):.2f}, reward={reward:.6f}")

        # Check for episode termination due to safety violations
        terminated = self.time >= self.max_episode_time
        
        # Early termination for severe constraint violations
        if abs(self.state[0]) > self.I_L_MAX * 1.2:  # 120% of current limit
            terminated = True
            reward -= 100.0  # Large penalty for unsafe operation

        truncated = False
        self.prev_error = error

        observation = np.array([vC, error, d_error, self.target_voltage], dtype=np.float32)
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = jnp.array([0.0, 0.0])
        self.prev_state = jnp.array([0.0, 0.0])
        self.time = 0.0
        self.prev_error = -self.target_voltage
        observation = np.array([0.0, -self.target_voltage, 0.0, self.target_voltage], dtype=np.float32)
        return observation, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass