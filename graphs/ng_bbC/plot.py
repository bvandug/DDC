#!/usr/bin/env python3
"""
buck_voltage_plotter_final.py

Loads Buck/BBC CSVs and creates a single SVG with
a main "Voltage vs Time" plot and a zoomed-in inset plot.

Files (expected in the current folder):
 - A2C_trajectory.csv
 - PPO_trajectory.csv
 - SAC_trajectory.csv
 - DDPG_trajectory.csv
 - DQN_all_episodes_trajectory.csv
 - TD3_all_episodes_trajectory.csv
 - plot_data_bbc_noise_0.0_goal_30.0V_ep_1.csv

Output:
 - BBC_voltage_vs_time_final.svg
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import List, Dict, Tuple, Union, Optional
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

def main():
    # Color + style map (colorblind-friendly)
    STYLE_MAP: Dict[str, Tuple[str, Union[str, Tuple[int, Tuple[int, ...]]]]] = {
        "DQN":  ("#005485", "-"),                 # blue, solid
        "TD3":  ("#D55E00", "--"),                # orange-brown, dashed
        "PPO":  ("#CC79A7", "-."),                # magenta, dash-dot
        "A2C":  ("#9467BD", ":"),                 # purple, dotted
        "SAC":  ("#E69F00", (0, (3, 1, 1, 1))),   # yellow-orange, dot-dash
        "DDPG": ("#009E0D", (0, (5, 2))),         # green, long dash
        "PID":  ("#000000", "-"),                 # black, solid
    }

    FILES: List[Path] = [
        Path("A2C_trajectory.csv"),
        Path("PPO_trajectory.csv"),
        Path("SAC_trajectory.csv"),
        Path("DDPG_trajectory.csv"),
        Path("DQN_all_episodes_trajectory.csv"),
        Path("TD3_all_episodes_trajectory.csv"),
        Path("plot_data_bbc_noise_0.0_goal_30.0V_ep_1.csv"),
    ]
    OUTPUT = "BBC_voltage_vs_time_final.svg"

    # Error band settings (target negative on BBC)
    TARGET_V = -30.0
    BAND_V = 1.2  # ±1.2 V

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'legend.fontsize': 13,
        'axes.titleweight': 'bold',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black',
        'legend.labelcolor': 'black',
    })

    fig, ax = plt.subplots(figsize=(12, 7))

    # hold all series for re-use in inset
    series: Dict[str, Dict[str, Union[str, np.ndarray]]] = {}

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

        # Clean + sort by time
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        order = np.argsort(x)
        x, y = x[order], y[order]

        lname = p.name.lower()
        if "dqn" in lname:
            label = "DQN"
        elif "td3" in lname:
            label = "TD3"
        elif "ppo" in lname:
            label = "PPO"
        elif "a2c" in lname:
            label = "A2C"
        elif "sac" in lname:
            label = "SAC"
        elif "ddpg" in lname:
            label = "DDPG"
        else:
            label = "PID"

        color, linestyle = STYLE_MAP.get(label, ("black", "-"))

        # Save for inset reuse
        series[p.stem] = {"x": x, "y": y, "label": label, "color": color, "linestyle": linestyle}

        # Main plot
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.6)

    # Error band lines
    ax.axhline(TARGET_V + BAND_V, color="0.3", linestyle="-", linewidth=0.9, zorder=0,
               label=f"{TARGET_V:.0f} V ± {BAND_V:.1f} V error band")
    ax.axhline(TARGET_V - BAND_V, color="0.3", linestyle="-", linewidth=0.9, zorder=0, label="_nolegend_")
    ax.axhline(-30, color="0.4", linestyle="-", linewidth=0.9, zorder=0, label="_nolegend_")
    
    ax.fill_between(
    x=[0, 0.05],              # match your x-axis range
    y1=-31.2, y2=-28.8,      
    color="grey", 
    alpha=0.15,             # transparency for subtle shading
    zorder=0
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, which="major", color="0.5", linewidth=0.4, alpha=0.4)
    ax.set_xlim(0, 0.05)

    # Deduplicate legend entries by label
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            uniq_handles.append(h); uniq_labels.append(l); seen.add(l)
    ax.legend(uniq_handles, uniq_labels, loc="upper right")

    # === Inset (FIXED: use stored data & styles, not last-loop vars) ===
    ax_inset = inset_axes(ax, width="35%", height="35%", loc="lower right", borderpad=1.7)
    for d in series.values():
        ax_inset.plot(
            d["x"], d["y"],
            label=d["label"],
            color=d["color"],
            linestyle=d["linestyle"],
            linewidth=1.6,
        )

    ax_inset.axhline(TARGET_V + BAND_V, color="0.3", linestyle="-", linewidth=0.8, zorder=0, label="_nolegend_")
    ax_inset.axhline(TARGET_V - BAND_V, color="0.3", linestyle="-", linewidth=0.8, zorder=0, label="_nolegend_")
    ax_inset.axhline(TARGET_V, color="0.3", linestyle="-", linewidth=0.8, zorder=0, label="_nolegend_")
    ax_inset.fill_between(
    x=[0, 0.05],              # match your x-axis range
    y1=-31.2, y2=-28.8,      
    color="grey", 
    alpha=0.15,             # transparency for subtle shading
    zorder=0
    )

    ax_inset.set_xlim(0.04, 0.05)
    ax_inset.set_ylim(-28.5, -33.5)
    ax_inset.invert_yaxis()  # mirror vertically

    ax_inset.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_inset.patch.set_alpha(0.95)
    for s in ax_inset.spines.values():
        s.set_linewidth(1.0)
    # No legend in inset
    leg = ax_inset.legend()
    if leg:
        leg.remove()

    plt.savefig(OUTPUT, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT} with main plot and inset plot.")

if __name__ == "__main__":
    main()
