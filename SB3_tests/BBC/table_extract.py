# json_compiler.py
# Scans the 'eval_runs' directory for all 'summary_metrics.json' files,
# consolidates the data, and outputs a summary table and an Excel report.

import os
import json
from pathlib import Path
import pandas as pd

# -------- Configuration --------
ROOT_DIR = Path("eval_runs")            # The root folder to scan for evaluation results.
OUTPUT_EXCEL_FILE = Path("bbc_summary_metrics.xlsx")

def collect_data_rows():
    """
    Recursively finds all 'summary_metrics.json' files and parses them.
    Crucially, it infers the algorithm, training noise, and evaluation noise
    from the file path for accuracy.
    
    Returns:
        list: A list of dictionaries, where each dictionary represents a row of data.
    """
    data_rows = []
    print(f"🔍 Scanning for summary files in '{ROOT_DIR}'...")

    for json_path in ROOT_DIR.rglob("summary_metrics.json"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # --- Infer context from the path, which is more reliable ---
            # This logic assumes a structure like: ROOT_DIR/ALGO/EVAL_CONDITION/RUN_NAME/summary.json
            inferred_algo = None
            inferred_training_noise = 0.0
            inferred_eval_noise = 0.0

            # The path is split into parts. We check the parent directories.
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
                except (ValueError, IndexError): pass
                
                # Infer Evaluation Noise from the evaluation condition folder
                try:
                    inferred_eval_noise = float(eval_condition_part.split('eval_noise_')[-1])
                except (ValueError, IndexError): pass

            # Overwrite the data from the JSON with our inferred, more reliable values.
            data['algo'] = inferred_algo
            data['training_noise_std'] = inferred_training_noise
            data['evaluation_noise_std'] = inferred_eval_noise
            
            data_rows.append(data)

        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not parse JSON file: {json_path}")
        except Exception as e:
            print(f"An unexpected error occurred with file {json_path}: {e}")
            
    return data_rows

def write_excel_report(df: pd.DataFrame, output_path: Path):
    """
    Writes the DataFrame to a multi-sheet Excel file with auto-sized columns.
    
    Args:
        df (pd.DataFrame): The consolidated DataFrame of all results.
        output_path (Path): The path to save the Excel file.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Results", index=False)
        
        if 'Algorithm' in df.columns:
            for algo_name, group_df in df.groupby("Algorithm"):
                sheet_name = str(algo_name)[:31]
                group_df.to_excel(writer, sheet_name=sheet_name, index=False)

        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns, 1):
                max_len = max(len(str(col)), df[col].astype(str).map(len).max())
                worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = max_len + 2

    print(f"\n✅ Successfully wrote {len(df)} rows to '{output_path}'")
    print("   - Includes an 'All Results' sheet and a separate sheet for each algorithm.")

def main():
    """
    Main function to orchestrate the data collection and report generation.
    """
    if not ROOT_DIR.exists():
        print(f"❌ Error: Root directory '{ROOT_DIR}' not found. Please run the evaluation script first.")
        return

    rows = collect_data_rows()
    if not rows:
        print("No 'summary_metrics.json' files were found. Nothing to process.")
        return
        
    print(f"Found {len(rows)} summary files.")
    df = pd.DataFrame(rows)

    # --- Data Cleaning and Structuring ---
    # EDIT THIS LIST to control which columns appear in the final report.
    column_order = [
        'algo',
        'training_noise_std',
        'evaluation_noise_std',
        'mean_reward',
        # 'std_reward',
        'mean_stabilisation_time_s',
        'mean_steady_state_error_v',
        'mean_overshoot_v',
        # 'mean_undershoot_v',
        'run_name',
        # 'episodes',
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
        'std_reward': 'Std Reward',
        'mean_stabilisation_time_s': 'Stabilisation Time (s)',
        'mean_steady_state_error_v': 'Steady State Error (V)',
        'mean_overshoot_v': 'Overshoot (V)',
        'mean_undershoot_v': 'Undershoot (V)',
    }, inplace=True)

    # --- Output Generation ---
    print("\n--- Consolidated Performance Metrics ---")
    print(df.to_string(index=False))
    write_excel_report(df, OUTPUT_EXCEL_FILE)

if __name__ == "__main__":
    main()

