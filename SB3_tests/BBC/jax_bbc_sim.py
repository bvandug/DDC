import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt

# --- Buck-Boost Parameters from Simulink ---
Vin = 48.0                   # Input voltage [V]
L = 220e-6                   # Inductance [H]
C = 100e-6                   # Capacitance [F]
R = 5.1                      # Load resistance [Ohm]
Ron_switch = 0.1            # MOSFET on-resistance [Ohm]
Ron_diode = 0.01            # Diode resistance [Ohm]
Vf_diode = 0.0              # Diode forward voltage [V]

D = 0.45                    # Duty cycle
fsw = 10e3                  # Switching frequency [Hz]
Tsw = 1 / fsw               # Switching period
sim_time = 10e-3            # Total simulation time [s]
dt = 5e-6                   # Time step [s]

# Time vector
t = jnp.arange(0, sim_time, dt)

# PWM signal: returns 1 (switch ON) or 0 (OFF)
@jit
def pwm(t):
    return jnp.where((t % Tsw) < D * Tsw, 1.0, 0.0)

# Buck-Boost converter dynamics
@jit
def buck_boost_dynamics(x, t):
    iL, vC = x
    u = pwm(t)

    # When the switch is ON
    def on_state(_):
        diL = (Vin - iL * Ron_switch) / L
        dvC = -vC / (R * C)
        return jnp.array([diL, dvC])

    # When the switch is OFF
    def off_state(_):
        diL = (-vC - iL * Ron_diode - Vf_diode) / L
        dvC = (iL - vC / R) / C
        return jnp.array([diL, dvC])

    return lax.cond(u == 1.0, on_state, off_state, operand=None)

# Initial state [iL, vC]
x0 = jnp.array([0.0, 0.0])

# Solve
sol = odeint(buck_boost_dynamics, x0, t)
iL, vC = sol[:, 0], sol[:, 1]

# Plot results
plt.plot(t * 1e3, vC, label="Output Voltage $v_C$ [V]")
plt.plot(t * 1e3, iL, label="Inductor Current $i_L$ [A]")
plt.xlabel("Time [ms]")
plt.title("Buck-Boost Converter Simulation (JAX)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
