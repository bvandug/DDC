#!/usr/bin/env python3
"""
angle_vs_time_final.py

Assumes EVERY CSV has the exact columns:
    time_s,angle_rad

Files (expected in the current folder):
 - dqn_cartpole_trace.csv
 - td3_cartpole_trace.csv
 - PID_clamped_angle.csv

Output:
 - angle_vs_time_final.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict

def main():
    FILES: List[Path] = [
        Path("dqn_cartpole_trace.csv"),
        Path("td3_cartpole_trace.csv"),
        Path("PID_clamped_angle.csv"),
    ]
    OUTPUT = "angle_vs_time_final.pdf"

    # Optional styling
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'legend.fontsize': 16,
        'axes.titleweight': 'bold',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black',
        'legend.labelcolor': 'black',
    })

    # Fixed colors: DQN=red, TD3=blue, PID=orange
    def color_for(label: str):
        u = label.upper()
        if u == "DQN":
            return "blue"
        if u == "TD3":
            return "red"
        if u == "PID":
            return "green"
        return None  # others -> default cycle

    fig, ax = plt.subplots(figsize=(12, 7))

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for p in FILES:
        if not p.exists():
            print(f"[WARN] Missing file: {p}")
            continue

        df = pd.read_csv(p, usecols=["time_s", "angle_rad"])
        x = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["angle_rad"], errors="coerce").to_numpy()

        # Clean/sort
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        order = np.argsort(x)
        x, y = x[order], y[order]

        # Label from filename
        lname = p.name.lower()
        if "dqn" in lname:
            label = "DQN"
        elif "td3" in lname:
            label = "TD3"
        elif "pid" in lname:
            label = "PID"
        else:
            label = p.stem

        data[label] = {"x": x, "y": y}
        ax.plot(x, y, label=label, color=color_for(label), linewidth=1.6)

    # Dashed error band lines at ±0.01 rad with a single legend entry
    ax.axhline(+0.1, color="0.3", linestyle="--", linewidth=0.9, zorder=0, label="±0.01 error band")
    ax.axhline(-0.1, color="0.3", linestyle="--", linewidth=0.9, zorder=0, label="_nolegend_")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.grid(True, which="major", color="0.85", linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper right")

    plt.savefig(OUTPUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT}.")

if __name__ == "__main__":
    main()