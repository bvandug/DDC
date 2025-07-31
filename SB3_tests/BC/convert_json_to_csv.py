import json
import pandas as pd
import os
import sys # To check for colab environment

# --- Check if running in Google Colab and import drive ---
IS_COLAB = True
if IS_COLAB:
    from google.colab import drive

def convert_json_to_csv(json_path="all_models_evaluation_data.json", csv_path="evaluation_data_for_excel.csv"):
    """
    Reads the evaluation data from the JSON file, processes it, and saves it
    as a CSV file formatted for easy plotting in Excel.
    Now supports saving to Google Drive when run in Colab.

    Args:
        json_path (str): The path to the input JSON file.
        csv_path (str): The path where the output CSV file will be saved.
    """
    print(f"--- Loading data from {json_path} ---")
    
    # Check if the JSON file exists
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON file not found at '{json_path}'. Please make sure the file is in the same directory.")
        return

    try:
        with open(json_path, 'r') as f:
            all_models_data = json.load(f)
    except json.JSONDecodeError:
        print(f"[ERROR] Could not decode JSON from '{json_path}'. The file may be corrupted or empty.")
        return

    print("--- Processing data for all models ---")
    
    # Use a list to collect DataFrames for each model
    all_dfs = []

    for model_name, model_data in all_models_data.items():
        # The data is nested: model_data is a list containing one episode list
        # The episode list contains [times, voltages, goal]
        if not model_data or not isinstance(model_data[0], list) or len(model_data[0]) < 2:
            print(f"[WARNING] Skipping {model_name} due to unexpected data format.")
            continue
            
        episode = model_data[0]
        times = episode[0]
        voltages = episode[1]
        
        # Create a DataFrame for the current model
        df = pd.DataFrame({
            f'{model_name}_Time (s)': times,
            f'{model_name}_Voltage (V)': voltages
        })
        all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No valid data was processed. CSV file will not be created.")
        return

    # Concatenate all individual DataFrames side-by-side
    # This works well because pandas aligns on the index. If some runs
    # have different numbers of steps, NaN values will be used for padding.
    combined_df = pd.concat(all_dfs, axis=1)

    # Save the final DataFrame to a CSV file
    try:
        # --- MODIFIED: Added 'sep' and 'decimal' arguments for Excel compatibility ---
        combined_df.to_csv(csv_path, index=False, sep=';', decimal=',')
        # ---
        print(f"\n--- Successfully converted data to {csv_path} ---")
        print("You can now open this file in Excel to plot your graphs.")
    except Exception as e:
        print(f"\n[ERROR] Failed to save CSV file. Reason: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    INPUT_JSON_FILE = "all_models_evaluation_data.json"
    OUTPUT_CSV_FILE = "evaluation_data_for_excel.csv"

    # --- Path Handling for Google Drive ---
    if IS_COLAB:
        print("Running in Google Colab. Mounting Google Drive...")
        # Define a base path in your Google Drive
        drive_save_path = "/content/drive/MyDrive/DDC/Combined_Results/"
        os.makedirs(drive_save_path, exist_ok=True)
        # Prepend the Drive path to the output filename
        final_csv_path = os.path.join(drive_save_path, OUTPUT_CSV_FILE)
    else:
        # If not in Colab, save to the local directory
        final_csv_path = OUTPUT_CSV_FILE
    
    # Run the conversion
    convert_json_to_csv(json_path=INPUT_JSON_FILE, csv_path=final_csv_path)
