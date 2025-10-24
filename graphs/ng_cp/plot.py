#!/usr/bin/env python3
"""
angle_vs_time_final.py

Assumes EVERY CSV has the exact columns:
    time_s,angle_rad

Files (expected in the current folder):
 - PPO_episode_1_trace.csv
 - A2C_episode_1_trace.csv
 - SAC_episode_1_trace.csv
 - DDPG_episode_1_trace.csv
 - dqn_cartpole_trace.csv
 - td3_cartpole_trace.csv
 - PID_clamped_angle.csv

Output:
 - angle_vs_time_final.svg
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union

def main():
    FILES: List[Path] = [
        Path("A2C_episode_1_trace.csv"),
        Path("PPO_episode_1_trace.csv"),
        Path("SAC_episode_1_trace.csv"),
        Path("DDPG_episode_1_trace.csv"),
        Path("dqn_cartpole_trace.csv"),
        Path("td3_cartpole_trace.csv"),
        Path("PID_clamped_angle.csv"),
    ]
    OUTPUT = "angle_vs_time_final.svg"

    # Publication styling
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "axes.titleweight": "bold",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
        "legend.labelcolor": "black",
    })

    # Color + style map (colorblind-friendly)
    STYLE_MAP: Dict[str, Tuple[str, Union[str, Tuple[int, Tuple[int, ...]]]]] = {
        "DQN":  ("#005485", "-"),                 # blue, solid
        "TD3":  ("#D55E00", "--"),                # red, dashed
        "PPO":  ("#CC79A7", "-."),                # orange, dash-dot
        "A2C":  ("#9467BD", ":"),                 # purple, dotted
        "SAC":  ("#E69F00", (0, (3, 1, 1, 1))),   # cyan, dot-dash
        "DDPG": ("#009E0D", (0, (5, 2))),         # pink-magenta, long dash
        "PID":  ("#000000", "-"),                 # dark grey, solid
    }

    def infer_label_from_name(fname: str) -> str:
        f = fname.lower()
        if "dqn" in f:  return "DQN"
        if "td3" in f:  return "TD3"
        if "ppo" in f:  return "PPO"
        if "a2c" in f:  return "A2C"
        if "sac" in f:  return "SAC"
        if "ddpg" in f: return "DDPG"
        return "PID"

    fig, ax = plt.subplots(figsize=(12, 7))

    for p in FILES:
        if not p.exists():
            print(f"[WARN] Missing file: {p}")
            continue

        df = pd.read_csv(p, usecols=["time_s", "angle_rad"])
        x = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["angle_rad"], errors="coerce").to_numpy()

        # Clean and sort
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        order = np.argsort(x)
        x, y = x[order], y[order]

        label = infer_label_from_name(p.name)
        color, linestyle = STYLE_MAP.get(label, ("black", "-"))
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.8)

    # ±0.1 rad error band
    ax.axhline(+0.1, color="0.4", linestyle="-", linewidth=0.9, zorder=0, label="±0.1 error band")
    ax.axhline(-0.1, color="0.4", linestyle="-", linewidth=0.9, zorder=0, label="_nolegend_")
    ax.axhline(0, color="0.4", linestyle="-", linewidth=0.9, zorder=0, label="_nolegend_")

    ax.fill_between(
    x=[0, 5],              # match your x-axis range
    y1=-0.1, y2=+0.1,      
    color="grey", 
    alpha=0.15,             # transparency for subtle shading
    zorder=0
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.set_xlim(0,5)
    ax.grid(True, which="major", color="0.85", linewidth=0.6, alpha=0.6)

    leg = ax.legend(loc="lower right", frameon=True, framealpha=0.9, borderpad=0.6)
    for lh in leg.legend_handles:
        lh.set_linewidth(2.2)

    plt.savefig(OUTPUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT}.")

if __name__ == "__main__":
    main()
