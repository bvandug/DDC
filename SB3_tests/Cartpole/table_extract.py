#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
table_extract.py
Scans result text files for Summary Metrics and builds a consolidated table.

- Keeps the same metric names as in the Summary (excluding the theoretical % line).
- Adds Algorithm / EnvNoise / RunNoise columns inferred from folder names.
- Prints a pretty table to stdout.
- Writes an Excel workbook (summary_metrics.xlsx) with an "All" sheet and one sheet per algorithm.

Folder layout expectation (examples):
plots/<Algorithm>/.../_env_noise_<float>/.../_noise_<float>/.../*_results.txt
"""
import os
import re
from pathlib import Path
import pandas as pd

# -------- Configuration --------
ROOT = Path("plots")                 # Root folder to scan
OUT_XLSX = Path("summary_metrics.xlsx")

# -------- Regexes for ALL summary metrics (robust to spacing/case/units) --------
RE_MEAN_REWARD = re.compile(r"^Mean\s+Reward:\s*([0-9.]+)", re.MULTILINE | re.IGNORECASE)

RE_MEAN_STAB = re.compile(
    r"^Mean\s+Stabilisation\s+Time\s*:\s*([0-9.]+)\s*s",
    re.MULTILINE | re.IGNORECASE,
)

RE_MEAN_OFFSET = re.compile(
    r"^Mean\s+Steady-state\s+offset\s*:\s*([-0-9.]+)\s*°",
    re.MULTILINE | re.IGNORECASE,
)

RE_MEAN_STABLE = re.compile(
    r"^Mean\s+Total\s+Stable\s+Time\s*:\s*([0-9.]+)\s*s",
    re.MULTILINE | re.IGNORECASE,
)

RE_MEAN_MAE = re.compile(
    r"^Mean\s+Steady-?State\s+MAE\s*:\s*([0-9.]+)\s*°",
    re.MULTILINE | re.IGNORECASE,
)

RE_MEAN_RMSE = re.compile(
    r"^Mean\s+Steady-?State\s+RMSE\s*:\s*([0-9.]+)\s*°",
    re.MULTILINE | re.IGNORECASE,
)

RE_MEAN_SIGMA = re.compile(
    r"^Mean\s+Oscillation\s*\(.*?\)\s*:\s*([0-9.]+)\s*°?",
    re.MULTILINE | re.IGNORECASE,
)

RE_MEAN_OVRSH = re.compile(
    r"^Mean\s*\|Overshoot/Undershoot\|\s*:\s*([0-9.]+)\s*°",
    re.MULTILINE | re.IGNORECASE,
)

RE_MEAN_IAE = re.compile(
    r"^Mean\s+IAE\s*:\s*([0-9.]+)",
    re.MULTILINE | re.IGNORECASE,
)

RE_MEAN_ENERGY = re.compile(
    r"^Mean\s+Control\s+Energy\s*:\s*([0-9.]+)",
    re.MULTILINE | re.IGNORECASE,
)

# -------- Helper functions --------
def extract_algo_env_run(path: Path):
    """
    Infer algorithm, env_noise and run_noise from folder names up the tree.
    - Algorithm: immediate child dir under ROOT
    - EnvNoise: any folder name containing "_env_noise_<float>"
    - RunNoise: any folder name containing "_noise_<float>"
    """
    algo = None
    env_noise = 0.0
    run_noise = 0.0

    # Algorithm = top-level under ROOT
    for p in path.parents:
        if p.parent.name == ROOT.name:
            algo = p.name
            break

    # Noise values from any ancestor folder
    for p in path.parents:
        name = p.name.lower()
        if "_env_noise_" in name:
            try:
                env_noise = float(name.split("_env_noise_")[1])
            except Exception:
                pass
        elif "_noise_" in name:
            try:
                run_noise = float(name.split("_noise_")[1])
            except Exception:
                pass

    return algo, env_noise, run_noise


def parse_results_text(text: str):
    """
    Parse all Summary Metrics from a single results text.
    Returns a dict with EXACT keys as your report (units removed from values).
    """
    def grab(rx):
        m = rx.search(text)
        return float(m.group(1)) if m else None

    return {
        # Keep EXACT names (as in your report)
        "Mean Reward": grab(RE_MEAN_REWARD),
        "Mean Stabilisation Time": grab(RE_MEAN_STAB),
        "Mean Steady-state offset": grab(RE_MEAN_OFFSET),
        "Mean Total Stable Time": grab(RE_MEAN_STABLE),
        "Mean Steady-State MAE": grab(RE_MEAN_MAE),
        "Mean Steady-State RMSE": grab(RE_MEAN_RMSE),
        "Mean Oscillation (σ)": grab(RE_MEAN_SIGMA),
        "Mean |Overshoot/Undershoot|": grab(RE_MEAN_OVRSH),
        "Mean IAE": grab(RE_MEAN_IAE),
        "Mean Control Energy": grab(RE_MEAN_ENERGY),
    }


def collect_rows():
    rows = []
    for results_path in ROOT.rglob("*_results.txt"):
        try:
            text = results_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        metrics = parse_results_text(text)
        # Skip files that don't have a "Mean Reward" (likely not a summary)
        if all(v is None for v in metrics.values()):
            continue

        algo, env_noise, run_noise = extract_algo_env_run(results_path)
        rows.append({
            "Algorithm": algo,
            "EnvNoise": env_noise,
            "RunNoise": run_noise,
            **metrics,
            "results_path": str(results_path),
        })
    return rows


def print_table(df: pd.DataFrame):
    # Order columns for readability
    preferred_order = [
        "Algorithm", "EnvNoise", "RunNoise",
        "Mean Reward",
        "Mean Stabilisation Time",
        "Mean Steady-state offset",
        "Mean Total Stable Time",
        "Mean Steady-State MAE",
        "Mean Steady-State RMSE",
        "Mean Oscillation (σ)",
        "Mean |Overshoot/Undershoot|",
        "Mean IAE",
        "Mean Control Energy",
    ]
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order and c != "results_path"]
    tdf = df[cols].copy()

    # Auto column widths based on string lengths
    col_w = []
    for c in tdf.columns:
        max_len = max(len(str(c)), *(len(str(v)) for v in tdf[c].astype(str).tolist() or [""]))
        col_w.append(max(8, min(max_len, 60)))

    # Header
    header_line = " | ".join(str(c).ljust(col_w[i]) for i, c in enumerate(tdf.columns))
    sep_line = "-+-".join("-" * col_w[i] for i in range(len(tdf.columns)))
    print(header_line)
    print(sep_line)

    # Rows
    for _, row in tdf.iterrows():
        print(" | ".join(str(row[c]).ljust(col_w[i]) for i, c in enumerate(tdf.columns)))


def main():
    if not ROOT.exists():
        print(f"[error] '{ROOT}' not found.")
        return

    rows = collect_rows()
    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)

    # Sort for consistency
    sort_cols = [c for c in ["Algorithm", "EnvNoise", "RunNoise"] if c in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)

    # Print table to console
    print_table(df)

    # Write Excel workbook
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xlw:
        # All rows
        df.to_excel(xlw, sheet_name="All", index=False)

        # One sheet per algorithm (sheet names limited to 31 chars)
        for algo, g in df.groupby("Algorithm", dropna=False):
            sheet = (str(algo) if algo is not None else "Unknown")[:31]
            g.to_excel(xlw, sheet_name=sheet, index=False)

        # Auto-size columns (first 1000 rows to bound work)
        for sheet_name, worksheet in xlw.sheets.items():
            data = df if sheet_name == "All" else df[df["Algorithm"] == sheet_name]
            for i, col in enumerate(data.columns, start=1):
                # compute a reasonable width
                sample_vals = data[col].astype(str).head(1000).tolist()
                max_len = max(len(str(col)), *(len(s) for s in sample_vals)) if sample_vals else len(str(col))
                width = min(max(10, max_len + 2), 60)
                worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = width

    print(f"\nWrote {len(df)} rows to {OUT_XLSX}")


if __name__ == "__main__":
    main()
