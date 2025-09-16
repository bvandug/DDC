#!/usr/bin/env python3
"""
buck_voltage_plotter_final.py

Loads three Buck CSVs and writes a single PDF with:
 - Main "Voltage vs Time" plot
 - Zoomed inset

Files (expected in the current folder):
 - evaluation_results_30.0_DQN_data.csv
 - evaluation_results_30.0_TD3_data.csv
 - PID_plot_data_noise_0.0_episode_1.csv

Output:
 - buck_voltage_vs_time_final.pdf
"""

from pathlib import Path
from typing import Optional, List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --------- Config ---------
FILES: List[Path] = [
    Path("evaluation_results_30.0_DQN_data.csv"),
    Path("evaluation_results_30.0_TD3_data.csv"),
    Path("PID_plot_data_noise_0.0_episode_1.csv"),
]
OUTPUT = "buck_voltage_vs_time_final.pdf"

TARGET_V = 30.0
BAND_V = 0.5            # ± band around target
DT = 5e-6               # seconds per step if only 'step' is present (adjust if needed)

TIME_CANDS = ["time_s", "time_ms", "time", "t", "timestamp", "t_s"]
STEP_CANDS = ["step", "global_step", "episode_step", "timestep", "time_step"]
VOLT_ALIASES = {"voltage_v", "voltage", "v", "vout", "v_out", "vo",
                "output voltage", "output_voltage", "vc", "vload"}

# Fixed colors for consistency
def color_for(label: str):
    u = label.upper()
    if u == "DQN": return "red"
    if u == "TD3": return "blue"
    if u == "PID": return "green"
    return None

def read_csv_any(path: Path) -> pd.DataFrame:
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
            # At least one numeric column? then accept
            if any(df.dtypes[c].kind in "ifc" for c in df.columns):
                return df
        except Exception:
            continue
    # Fallback to default
    return pd.read_csv(path)

def is_voltage_name(col: str) -> bool:
    s = col.strip().lower()
    s = re.sub(r"\(.*?\)", "", s).strip()
    if s in VOLT_ALIASES:
        return True
    if ("volt" in s) or (re.search(r"\bv\b", s) and "dv" not in s):
        return True
    if s.startswith("v") and " " not in s and "dv" not in s:
        return True
    if "(v)" in col.lower() or " [v]" in col.lower():
        return True
    return False

def detect_voltage_column(df: pd.DataFrame) -> Optional[str]:
    numeric = [c for c in df.columns if df[c].dtype.kind in "ifc"]
    cands = [c for c in numeric if is_voltage_name(c)]
    if cands:
        order = ["voltage_v", "vout", "vo", "output voltage", "output_voltage", "vc", "voltage", "v"]
        for key in order:
            for c in cands:
                if key == c.strip().lower():
                    return c
        return cands[0]
    # last-chance heuristic
    for c in numeric:
        if ("volt" in c.lower()) or ("v" in c.lower()):
            return c
    return None

def detect_time_or_step(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    # explicit time first
    for key in TIME_CANDS:
        if key in cols:
            return cols[key], None
    # step second
    for key in STEP_CANDS:
        if key in cols:
            return None, cols[key]
    # nothing found
    return None, None

def series_from_df(df: pd.DataFrame, dt_default: float):
    # Normalize potential number formats
    df = df.copy()
    # TIME / STEP detection
    time_col, step_col = detect_time_or_step(df)
    # VOLT detection
    volt_col = detect_voltage_column(df)
    if volt_col is None:
        raise ValueError(f"No voltage-like column found. Columns: {list(df.columns)}")

    # Build x (seconds)
    if time_col is not None:
        if time_col.lower() == "time_ms":
            x = pd.to_numeric(df[time_col], errors="coerce").to_numpy() / 1000.0
        else:
            x = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    elif step_col is not None:
        steps = pd.to_numeric(df[step_col], errors="coerce").to_numpy()
        x = steps * float(dt_default)
    else:
        # Fall back to index → seconds with dt_default
        idx = np.arange(len(df), dtype=float)
        x = idx * float(dt_default)

    y = pd.to_numeric(df[volt_col], errors="coerce").to_numpy()

    # Clean and sort
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        raise ValueError("All time/voltage values are NaN after parsing.")
    order = np.argsort(x)
    return x[order], y[order], volt_col

def main():
    # A bit of consistent styling
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
        'figure.dpi': 150
    })

    fig, ax = plt.subplots(figsize=(12, 7))
    data = {}

    for p in FILES:
        if not p.exists():
            print(f"[WARN] Skipping {p.name}: file not found.")
            continue
        try:
            df = read_csv_any(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: cannot read CSV ({e})")
            continue

        # Decide label based on filename
        lname = p.name.lower()
        if "dqn" in lname:
            label = "DQN"
        elif "td3" in lname:
            label = "TD3"
        else:
            label = "PID"

        try:
            x, y, vcol = series_from_df(df, DT)
            print(f"[INFO] {p.name}: label={label}, using voltage='{vcol}'")
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue

        data[p.stem] = {"x": x, "y": y, "label": label}
        ax.plot(x, y, label=label, color=color_for(label), linewidth=1.6)

    # Error band (two dashed lines, single legend entry)
    ax.axhline(TARGET_V + BAND_V, color="0.3", linestyle="--", linewidth=0.9,
               zorder=0, label=f"{TARGET_V:.0f} V ± {BAND_V:.1f} V")
    ax.axhline(TARGET_V - BAND_V, color="0.3", linestyle="--", linewidth=0.9,
               zorder=0, label="_nolegend_")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Buck Converter — Voltage vs Time")
    ax.grid(True, which="major", color="0.85", linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper right")

    # Inset
    ax_inset = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=2.5)
    for d in data.values():
        ax_inset.plot(d["x"], d["y"], color=color_for(d["label"]), linewidth=1.2)
    ax_inset.axhline(TARGET_V + BAND_V, color="0.3", linestyle="--", linewidth=0.8, zorder=0, label="_nolegend_")
    ax_inset.axhline(TARGET_V - BAND_V, color="0.3", linestyle="--", linewidth=0.8, zorder=0, label="_nolegend_")
    # Adjust zoom as you like:
    ax_inset.set_xlim(4e-4, 1e-3)
    ax_inset.set_ylim(20, 31)
    ax_inset.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_inset.patch.set_alpha(0.95)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(1.0)

    plt.savefig(OUTPUT, bbox_inches="tight")  # PDF because of .pdf extension
    plt.close()
    print(f"Saved {OUTPUT} with main plot and inset.")

if __name__ == "__main__":
    main()
