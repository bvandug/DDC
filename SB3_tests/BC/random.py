import os
import shutil
import sys
from google.colab import drive

# --- USER CONFIGURATION ---
# The name of the folder to be created in your Google Drive's "My Drive"
# Feel free to change this to whatever you like.
DRIVE_FOLDER_NAME = "BuckConverterTuningResults"

# The names of the local output directories created by your tuning script
LOCAL_LOG_DIR = "./buck_converter_tuning_logs/"
LOCAL_HYPERPARAM_DIR = "./hyperparameters_SSH/"

# List all the algorithm names to find their corresponding Optuna DB files.
ALGORITHMS_TO_TUNE = ["A2C", "PPO", "SAC", "TD3", "DDPG", "DQN"]

def save_to_drive():
    """
    Mounts Google Drive and saves all tuning output files.
    """
    print("="*80)
    print("Starting process to save tuning outputs to Google Drive...")
    print("="*80)

    # Define the target directory in Google Drive
    drive_output_path = os.path.join("/content/drive/MyDrive", DRIVE_FOLDER_NAME)
    os.makedirs(drive_output_path, exist_ok=True)
    print(f"Created/found Google Drive destination: {drive_output_path}")
    print("-" * 80)

    # 2. Save TensorBoard logs directory
    if os.path.exists(LOCAL_LOG_DIR):
        try:
            target_tb_path = os.path.join(drive_output_path, os.path.basename(LOCAL_LOG_DIR.strip('/')))
            if os.path.exists(target_tb_path):
                print(f"Removing existing TensorBoard logs in Drive: {target_tb_path}")
                shutil.rmtree(target_tb_path)
            shutil.copytree(LOCAL_LOG_DIR, target_tb_path)
            print(f"✅ TensorBoard logs saved to: {target_tb_path}")
        except Exception as e:
            print(f"❌ Error saving TensorBoard logs: {e}")
    else:
        print(f"⚠️ TensorBoard logs directory not found: {LOCAL_LOG_DIR}")
    print("-" * 80)

    # 3. Save hyperparameter JSON files directory
    if os.path.exists(LOCAL_HYPERPARAM_DIR):
        try:
            target_hp_path = os.path.join(drive_output_path, os.path.basename(LOCAL_HYPERPARAM_DIR.strip('/')))
            if os.path.exists(target_hp_path):
                print(f"Removing existing hyperparameter JSONs in Drive: {target_hp_path}")
                shutil.rmtree(target_hp_path)
            shutil.copytree(LOCAL_HYPERPARAM_DIR, target_hp_path)
            print(f"✅ Hyperparameter JSON files saved to: {target_hp_path}")
        except Exception as e:
            print(f"❌ Error saving hyperparameter JSONs: {e}")
    else:
        print(f"⚠️ Hyperparameter results directory not found: {LOCAL_HYPERPARAM_DIR}")
    print("-" * 80)

    # 4. Save Optuna database files
    db_files_saved = []
    for algorithm in ALGORITHMS_TO_TUNE:
        study_name = f"{algorithm}-bc-tuning-seed42"
        db_file = f"{study_name}.db"
        if os.path.exists(db_file):
            try:
                target_db_path = os.path.join(drive_output_path, db_file)
                shutil.copy(db_file, target_db_path)
                db_files_saved.append(db_file)
            except Exception as e:
                print(f"❌ Error saving Optuna database file '{db_file}': {e}")
    if db_files_saved:
        print(f"✅ Optuna database files saved: {', '.join(db_files_saved)}")
    else:
        print("⚠️ No Optuna database files found to save.")
    print("-" * 80)

    print("\n" + "="*80)
    print("All specified tuning outputs successfully saved to Google Drive!")
    print("="*80)

if __name__ == "__main__":
    save_to_drive()