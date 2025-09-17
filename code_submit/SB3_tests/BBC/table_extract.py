
"""Aggregate evaluation metrics from multiple Simulink runs.

Scans the `eval_simulink_runs` directory for `summary_metrics.json`
files, extracts and infers algorithm/run metadata, computes a
"Real Overshoot (V)" column from trajectory CSVs, prints a summary
table, and writes a multi-sheet Excel report.
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# -------- Configuration --------
ROOT_DIR = Path("eval_simulink_runs")            # The root folder to scan for evaluation results.
OUTPUT_EXCEL_FILE = Path("bbc_summary_metrics.xlsx")
CSV_NAME = "all_episodes_trajectory.csv"         # Must match evaluator output


def collect_data_rows() -> list[dict]:
    """Recursively find and parse all `summary_metrics.json` files.

    Infers algorithm name, training noise, and evaluation noise
    from directory structure, loads metrics from JSON, and attempts
    to compute a sign-aware real overshoot from a sibling CSV.

    Returns:
        list[dict]: One dictionary per run with consolidated metrics.
    """
    data_rows = []
    print(f" Scanning for summary files in '{ROOT_DIR}'...")

    for json_path in ROOT_DIR.rglob("summary_metrics.json"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # --- Infer context from the path (ROOT/ALGO/EVAL_CONDITION/RUN_NAME/summary.json) ---
            inferred_algo = None
            inferred_training_noise = 0.0
            inferred_eval_noise = 0.0
            run_name_part = None

            if len(json_path.parts) >= 4:
                run_name_part = json_path.parts[-2]          # e.g., 'td3_noise_0.100'
                eval_condition_part = json_path.parts[-3]    # e.g., 'td3_eval_noise_0.050'
                algo_part = json_path.parts[-4]              # e.g., 'TD3'

                # Infer Algorithm from the folder name
                if algo_part.upper() in ['TD3', 'DQN', 'A2C', 'SAC', 'PPO', 'DDPG']:
                    inferred_algo = algo_part.upper()

                # Infer Training Noise from the run name folder
                try:
                    inferred_training_noise = float(run_name_part.split('noise_')[-1])
                except (ValueError, IndexError):
                    pass
                
                # Infer Evaluation Noise from the evaluation condition folder
                try:
                    inferred_eval_noise = float(eval_condition_part.split('eval_noise_')[-1])
                except (ValueError, IndexError):
                    pass

            # Overwrite the data from the JSON with our inferred, more reliable values.
            data['algo'] = inferred_algo
            data['training_noise_std'] = inferred_training_noise
            data['evaluation_noise_std'] = inferred_eval_noise
            data['run_name'] = run_name_part

            # --- Compute "real overshoot" from CSV if available ---
            csv_path = json_path.parent / CSV_NAME
            data['real_overshoot_v'] = compute_real_overshoot_from_csv(csv_path)

            data_rows.append(data)

        except json.JSONDecodeError:
            print(f" Warning: Could not parse JSON file: {json_path}")
        except Exception as e:
            print(f"An unexpected error occurred with file {json_path}: {e}")
            
    return data_rows

def compute_real_overshoot_from_csv(csv_path: Path) -> float:
    """Compute sign-aware overshoot from per-episode CSV data.

    For each episode:
        - Take the last 10% of samples, compute the median voltage.
        - Estimate a global target as the median across episodes.
        - Compute overshoot relative to target:
            * If target < 0: use most negative dip beyond target.
            * If target > 0: use peak above target.
        - Average overshoot across all episodes.

    Args:
        csv_path (Path): Path to `all_episodes_trajectory.csv`.

    Returns:
        float: Mean overshoot (V) across episodes, or NaN if missing.
    """
    try:
        if not csv_path.exists():
            return float("nan")

        df = pd.read_csv(csv_path)
        required_cols = {"episode", "time_s", "voltage_v"}
        if not required_cols.issubset(df.columns):
            return float("nan")

        overshoots = []
        target_candidates = []

        for ep, g in df.groupby("episode"):
            g = g.sort_values("time_s")
            n = len(g)
            if n == 0:
                continue

            tail_n = max(1, int(0.1 * n))  # last 10%
            tail = g.iloc[-tail_n:]
            target_est_ep = float(tail["voltage_v"].median())
            target_candidates.append(target_est_ep)

        if len(target_candidates) == 0:
            return float("nan")

        target_est = float(np.median(target_candidates))
        sign = -1 if target_est < 0 else 1

        # Compute per-episode overshoot relative to target_est
        for ep, g in df.groupby("episode"):
            v = g["voltage_v"].to_numpy(dtype=float)
            if v.size == 0:
                continue

            if sign < 0:
                # Target negative: "overshoot" is the most negative dip beyond target (i.e., lower than target)
                ep_ov = max(0.0, target_est - float(np.min(v)))
            else:
                ep_ov = max(0.0, float(np.max(v)) - target_est)

            overshoots.append(ep_ov)

        if len(overshoots) == 0:
            return float("nan")

        return float(np.mean(overshoots))

    except Exception as e:
        print(f" compute_real_overshoot_from_csv error for {csv_path}: {e}")
        return float("nan")


def write_excel_report(df: pd.DataFrame, output_path: Path) -> None:
    """Write results DataFrame to a multi-sheet Excel file.

    Creates an "All Results" sheet and one sheet per algorithm.
    Auto-sizes columns for better readability.

    Args:
        df (pd.DataFrame): DataFrame containing summary metrics.
        output_path (Path): Path to save the Excel file.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Results", index=False)
        
        if 'Algorithm' in df.columns:
            for algo_name, group_df in df.groupby("Algorithm"):
                sheet_name = str(algo_name)[:31]
                group_df.to_excel(writer, sheet_name=sheet_name, index=False)

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            # Auto-size using current frame's columns
            for i, col in enumerate(df.columns, 1):
                max_len = max(len(str(col)), df[col].astype(str).map(len).max())
                worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = min(max_len + 2, 60)

    print(f"\n Successfully wrote {len(df)} rows to '{output_path}'")
    print("   - Includes an 'All Results' sheet and a separate sheet for each algorithm.")

def main():
    """Main orchestration function.

    Collects data, constructs a cleaned and sorted DataFrame,
    prints a summary table to stdout, and writes the Excel report.

    Raises:
        FileNotFoundError: If ROOT_DIR does not exist.
    """
    if not ROOT_DIR.exists():
        print(f"Error: Root directory '{ROOT_DIR}' not found. Please run the evaluation script first.")
        return

    rows = collect_data_rows()
    if not rows:
        print("No 'summary_metrics.json' files were found. Nothing to process.")
        return
        
    print(f"Found {len(rows)} summary files.")
    df = pd.DataFrame(rows)

    # --- Data Cleaning and Structuring ---
    column_order = [
        'algo',
        'training_noise_std',
        'evaluation_noise_std',
        'mean_reward',
        'mean_stabilisation_time_s',
        'mean_steady_state_error_v',
        # 'mean_overshoot_v',
        'real_overshoot_v',          # <--- new column
        # 'run_name',
    ]
    
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    sort_keys = [key for key in ['algo', 'training_noise_std', 'evaluation_noise_std'] if key in df.columns]
    if sort_keys:
        df.sort_values(by=sort_keys, inplace=True)
    
    df.rename(columns={
        'algo': 'Algorithm',
        'training_noise_std': 'Training Noise',
        'evaluation_noise_std': 'Evaluation Noise',
        'mean_reward': 'Mean Reward',
        'mean_stabilisation_time_s': 'Stabilisation Time (s)',
        'mean_steady_state_error_v': 'Steady State Error (V)',
        # 'mean_overshoot_v': 'Overshoot (V)',
        'real_overshoot_v': 'Real Overshoot (V)',
    }, inplace=True)

    # --- Output Generation ---
    print("\n--- Consolidated Performance Metrics ---")
    print(df.to_string(index=False))
    write_excel_report(df, OUTPUT_EXCEL_FILE)

if __name__ == "__main__":
    main()
