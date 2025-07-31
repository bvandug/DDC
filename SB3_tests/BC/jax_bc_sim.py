import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt

# --- System Parameters ---
Vin = 48.0           # Input voltage [V]
L = 100e-6           # Inductance [H]
C = 1000e-6          # Capacitance [F]
R = 10.0             # Load resistance [Ohm]
Ron_switch = 0.1     # MOSFET on-resistance [Ohm]
Ron_diode = 0.001    # Diode resistance [Ohm]
Vf_diode = 0.8       # Diode forward voltage [V]

D = 0.5              # PWM duty cycle
fsw = 10e3           # PWM switching frequency [Hz]
Tsw = 1 / fsw        # PWM period [s]
sim_time = 10e-3     # Total simulation duration [s]
dt = 5e-6            # Time step [s]

# --- Time Vector ---
t = jnp.arange(0, sim_time, dt)

# --- PWM Generator ---
@jit
def pwm(t):
    return jnp.where((t % Tsw) < D * Tsw, 1.0, 0.0)

# --- ODE System ---
@jit
def buck_dynamics(x, t):
    iL, vC = x
    u = pwm(t)

    vL = lax.cond(
        u == 1.0,
        lambda _: Vin - vC - iL * Ron_switch,
        lambda _: -vC - iL * Ron_diode - Vf_diode,
        operand=None
    )

    diL = vL / L
    dvC = (iL - vC / R) / C
    return jnp.array([diL, dvC])

# --- Initial Conditions ---
x0 = jnp.array([0.0, 0.0])

# --- Solve ---
sol = odeint(buck_dynamics, x0, t)
iL, vC = sol[:, 0], sol[:, 1]

# --- Plot ---
import matplotlib.pyplot as plt
plt.plot(t * 1e3, vC, label="Output Voltage $v_C$ [V]")
plt.plot(t * 1e3, iL, label="Inductor Current $i_L$ [A]")
plt.xlabel("Time [ms]")
plt.title("Buck Converter Simulation in JAX")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
