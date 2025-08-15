#!/usr/bin/env python3
import os
import re
from pathlib import Path
import pandas as pd

ROOT = Path("plots")  # change if needed
OUT_XLSX = Path("summary_metrics.xlsx")

RE_MEAN_REWARD = re.compile(r"^Mean Reward:\s*([0-9.]+)", re.MULTILINE)
RE_MEAN_STAB   = re.compile(r"^Mean\s+Stabilisation\s+Time\s*:\s*([0-9.]+)\s*s", re.MULTILINE | re.IGNORECASE)
RE_MEAN_OFFSET = re.compile(r"^Mean\s+Steady-state\s+offset\s*:\s*([-0-9.]+)\s*°", re.MULTILINE | re.IGNORECASE)
RE_MEAN_STABLE = re.compile(r"^Mean\s+Total\s+Stable\s+Time\s*:\s*([0-9.]+)\s*s", re.MULTILINE | re.IGNORECASE)

def extract_algo_env_run(path: Path):
    algo = None
    env_noise = 0.0  # default
    run_noise = 0.0  # default

    for p in path.parents:
        if p.parent.name == ROOT.name:
            algo = p.name
            break
    for p in path.parents:
        name = p.name.lower()
        if "_env_noise_" in name:
            try:
                env_noise = float(name.split("_env_noise_")[1])
            except:
                pass
        elif "_noise_" in name:   # run noise folder
            try:
                run_noise = float(name.split("_noise_")[1])
            except:
                pass
    return algo, env_noise, run_noise

def parse_results_text(text: str):
    def grab(rx):
        m = rx.search(text)
        return float(m.group(1)) if m else None
    return {
        "mean_reward": grab(RE_MEAN_REWARD),
        "mean_stabilisation_time_s": grab(RE_MEAN_STAB),
        "mean_steady_state_offset_deg": grab(RE_MEAN_OFFSET),
        "mean_total_stable_time_s": grab(RE_MEAN_STABLE),
    }

def collect_rows():
    rows = []
    for results_path in ROOT.rglob("*_results.txt"):
        try:
            text = results_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        metrics = parse_results_text(text)
        algo, env_noise, run_noise = extract_algo_env_run(results_path)
        rows.append({
            "algorithm": algo,
            "env_noise": env_noise,
            "run_noise": run_noise,
            **metrics,
            "results_path": str(results_path),
        })
    return rows

def print_table(df: pd.DataFrame):
    headers = [
        ("Algorithm", "algorithm"),
        ("EnvNoise", "env_noise"),
        ("RunNoise", "run_noise"),
        ("MeanReward", "mean_reward"),
        ("MeanStab(s)", "mean_stabilisation_time_s"),
        ("MeanOffset(°)", "mean_steady_state_offset_deg"),
        ("MeanStable(s)", "mean_total_stable_time_s"),
    ]
    col_w = [max(len(h[0]), df[h[1]].astype(str).map(len).max()) for h in headers]
    # Print header
    line = " | ".join(h[0].ljust(col_w[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * col_w[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for _, row in df.iterrows():
        print(" | ".join(str(row[h[1]]).ljust(col_w[i]) for i, h in enumerate(headers)))

def main():
    if not ROOT.exists():
        print(f"[error] '{ROOT}' not found.")
        return
    rows = collect_rows()
    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)
    df.sort_values(["algorithm","env_noise","run_noise"], inplace=True)

    # Print table to console
    print_table(df)

    # Write Excel
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xlw:
        df.to_excel(xlw, sheet_name="All", index=False)
        for algo, g in df.groupby("algorithm", dropna=False):
            sheet = (algo or "Unknown")[:31]
            g.to_excel(xlw, sheet_name=sheet, index=False)

        # Auto-size columns
        for sheet_name, worksheet in xlw.sheets.items():
            data = df if sheet_name == "All" else df[df["algorithm"] == sheet_name]
            for i, col in enumerate(data.columns, start=1):
                max_len = max(len(str(col)), *(len(str(v)) for v in data[col].head(1000)))
                worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = min(max_len + 2, 60)

    print(f"\nWrote {len(df)} rows to {OUT_XLSX}")

if __name__ == "__main__":
    main()
