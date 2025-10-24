#!/usr/bin/env python3
"""
buck_voltage_plotter_final.py

Loads the three known Buck CSVs and creates a single PDF with
a main "Voltage vs Time" plot and a zoomed-in inset plot.

Files (expected in the current folder):
 - evaluation_results_30.0_DQN_data.csv
 - evaluation_results_30.0_TD3_data.csv
 - PID_plot_data_noise_0.0_episode_1.csv

Output:
 - buck_voltage_vs_time_final.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import Optional, List
from pathlib import Path
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def read_csv_any(path: Path) -> pd.DataFrame:
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            if any(df.dtypes[k].kind in "ifc" for k in df.columns):
                return df
        except Exception:
            continue
    return pd.read_csv(path)

def is_voltage_name(col: str) -> bool:
    s = col.strip().lower()
    s = re.sub(r"\(.*?\)", "", s).strip()
    aliases = {"voltage_v", "voltage", "v", "vout", "v_out", "vo",
               "output voltage", "output_voltage", "vc", "vload"}
    if s in aliases:
        return True
    if ("volt" in s) or (re.search(r"\bv\b", s) and "dv" not in s):
        return True
    if s.startswith("v") and " " not in s and "dv" not in s:
        return True
    if "(v)" in col.lower() or " [v]" in col.lower():
        return True
    return False

def detect_voltage_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if df[c].dtype.kind in "ifc" and is_voltage_name(c)]
    if candidates:
        order = ["voltage_v", "vout", "vo", "output voltage", "output_voltage", "vc", "voltage", "v"]
        for key in order:
            for c in candidates:
                if key == c.strip().lower():
                    return c
        return candidates[0]
    for c in df.columns:
        if df[c].dtype.kind in "ifc" and (("volt" in c.lower()) or ("v" in c.lower())):
            return c
    return None

# Fixed colors: DQN=red, TD3=blue, PID=green (change to "orange" if you prefer)
def color_for(label: str):
    u = label.upper()
    if u == "DQN":
        return "blue"
    if u == "TD3":
        return "red"
    if u == "PID":
        return "green"
    return None

def main():
    FILES: List[Path] = [
        Path("DQN_all_episodes_trajectory.csv"),
        Path("TD3_all_episodes_trajectory.csv"),
        Path("plot_data_bbc_noise_0.0_goal_30.0V_ep_1.csv"),
        
    ]
    OUTPUT = "BBC_voltage_vs_time_final.pdf"

    # Error band settings
    TARGET_V = -30.0
    BAND_V = 1.2  # ±0.5 V around TARGET_V

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

    fig, ax = plt.subplots(figsize=(12, 7))

    data = {}
    for p in FILES:
        try:
            df = read_csv_any(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: cannot read CSV ({e})")
            continue

        volt_col = detect_voltage_column(df)
        if volt_col is None:
            print(f"[WARN] Skipping {p.name}: no voltage-like column detected.")
            continue

        if "time_ms" in df.columns:
            x = pd.to_numeric(df["time_ms"], errors="coerce").to_numpy() / 1000.0
        elif "time_s" in df.columns:
            x = pd.to_numeric(df["time_s"], errors="coerce").to_numpy()
        else:
            x = np.arange(len(df), dtype=float)

        y = pd.to_numeric(df[volt_col], errors="coerce").to_numpy()

        # Always sort by time
        order = np.argsort(x)
        x, y = x[order], y[order]

        lname = p.name.lower()
        if "dqn" in lname:
            label = "DQN"
        elif "td3" in lname:
            label = "TD3"
        else:
            label = "PID"

        data[p.stem] = {"x": x, "y": y, "label": label}
        ax.plot(x, y, label=label, color=color_for(label), linewidth=1.4)

    # Dashed error band lines at 30 V ± 0.5 V (single legend entry)
    ax.axhline(TARGET_V + BAND_V, color="0.3", linestyle="--", linewidth=0.9, zorder=0,
               label=f"{TARGET_V:.0f} V ± {BAND_V:.1f} V error band")
    ax.axhline(TARGET_V - BAND_V, color="0.3", linestyle="--", linewidth=0.9, zorder=0, label="_nolegend_")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, which="major", color="0.85", linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 0.1)
    

    # Inset with matching colors + same error band
    ax_inset = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=2.5)
    for d in data.values():
        ax_inset.plot(d["x"], d["y"], color=color_for(d["label"]), linewidth=1.2, label=d["label"])
    ax_inset.axhline(TARGET_V + BAND_V, color="0.3", linestyle="--", linewidth=0.8, zorder=0, label="_nolegend_")
    ax_inset.axhline(TARGET_V - BAND_V, color="0.3", linestyle="--", linewidth=0.8, zorder=0, label="_nolegend_")

    ax_inset.set_xlim(0.04, 0.05)
    ax_inset.set_ylim(-28.5, -33.5)
    ax_inset.invert_yaxis()  # flips the y-axis so it's mirrored vertically

    ax_inset.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_inset.patch.set_alpha(0.95)
    for s in ax_inset.spines.values():
        s.set_linewidth(1.0)
    ax_inset.set_title("")
    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")
    leg = ax_inset.legend()
    if leg:
        leg.remove()

    plt.savefig(OUTPUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT} with main plot and inset plot.")

if __name__ == "__main__":
    main()