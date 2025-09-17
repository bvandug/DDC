#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Read sim‐vs‐NumPy‐vs‐JAX CSV and save SVGs in degrees and radians."
    )
    parser.add_argument(
        "csvfile",
        help="Path to the cleaned CSV file (e.g. svg_clean.csv)"
    )
    args = parser.parse_args()

    # 1) Read the CSV
    df = pd.read_csv(args.csvfile)

    # 2) Extract by name
    t             = df["Time (s)"].to_numpy()
    sim_theta_deg = df["Simulink θ (°)"].to_numpy()
    np_theta_deg  = df["NumPy θ (°)"].to_numpy()
    jax_theta_deg = df["JAX θ (°)"].to_numpy()

    sim_thdot_deg = df["Simulink θ̇ (°/s)"].to_numpy()
    np_thdot_deg  = df["NumPy θ̇ (°/s)"].to_numpy()
    jax_thdot_deg = df["JAX θ̇ (°/s)"].to_numpy()

    # 3) Convert to radians
    sim_theta_rad   = np.deg2rad(sim_theta_deg)
    np_theta_rad    = np.deg2rad(np_theta_deg)
    jax_theta_rad   = np.deg2rad(jax_theta_deg)

    sim_thdot_rad   = np.deg2rad(sim_thdot_deg)
    np_thdot_rad    = np.deg2rad(np_thdot_deg)
    jax_thdot_rad   = np.deg2rad(jax_thdot_deg)

    # 4) Compute abs diffs (rad)
    abs_diff_np  = np.abs(sim_theta_rad - np_theta_rad)
    abs_diff_jax = np.abs(sim_theta_rad - jax_theta_rad)

    # —— Plot 1: θ in degrees ——
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, sim_theta_deg, label="Simulink")
    ax.plot(t, np_theta_deg,   "--", label="NumPy")
    ax.plot(t, jax_theta_deg,  ":",  label="JAX")
    ax.set(title="θ (°) vs Time", xlabel="Time (s)", ylabel="θ (°)")
    ax.legend(loc="upper left"); ax.grid(True)
    fig.tight_layout()
    fig.savefig("theta_deg.svg", format="svg")
    plt.close(fig)

    # —— Plot 2: θ in radians ——
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, sim_theta_rad, label="Simulink")
    ax.plot(t, np_theta_rad,   "--", label="NumPy")
    ax.plot(t, jax_theta_rad,  ":",  label="JAX")
    ax.set(title="θ (rad) vs Time", xlabel="Time (s)", ylabel="θ (rad)")
    ax.legend(loc="upper left"); ax.grid(True)
    fig.tight_layout()
    fig.savefig("theta_rad.svg", format="svg")
    plt.close(fig)

    # —— Plot 3: θ̇ in degrees/s ——
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, sim_thdot_deg, label="Simulink")
    ax.plot(t, np_thdot_deg,   "--", label="NumPy")
    ax.plot(t, jax_thdot_deg,  ":",  label="JAX")
    ax.set(title="θ̇ (°/s) vs Time", xlabel="Time (s)", ylabel="θ̇ (°/s)")
    ax.legend(loc="upper left"); ax.grid(True)
    fig.tight_layout()
    fig.savefig("theta_dot_deg.svg", format="svg")
    plt.close(fig)

    # —— Plot 4: θ̇ in radians/s ——
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, sim_thdot_rad, label="Simulink")
    ax.plot(t, np_thdot_rad,   "--", label="NumPy")
    ax.plot(t, jax_thdot_rad,  ":",  label="JAX")
    ax.set(title="θ̇ (rad/s) vs Time", xlabel="Time (s)", ylabel="θ̇ (rad/s)")
    ax.legend(loc="upper left"); ax.grid(True)
    fig.tight_layout()
    fig.savefig("theta_dot_rad.svg", format="svg")
    plt.close(fig)

    # —— Plot 5: abs diff (rad), log scale ——
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(t, abs_diff_np,  "--", label="|Sim – NumPy|")
    ax.semilogy(t, abs_diff_jax, ":",  label="|Sim – JAX|")
    ax.set(title="Absolute difference (rad)", xlabel="Time (s)", ylabel="|Δθ| (rad)")
    ax.legend(loc="upper left"); ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig("abs_diff.svg", format="svg")
    plt.close(fig)

    print("✅ SVGs written:\n"
          "  • theta_deg.svg\n"
          "  • theta_rad.svg\n"
          "  • theta_dot_deg.svg\n"
          "  • theta_dot_rad.svg\n"
          "  • abs_diff.svg")

if __name__ == "__main__":
    main()
