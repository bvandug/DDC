import subprocess
import sys
import os

# --- Configuration ---
# Define the algorithms and noise levels for the experiment.
ALGORITHMS_TO_RUN = ["td3", "dqn"]
# Training noise levels are handled by the '--noise all' flag in the training script.
# Evaluation noise levels need to be specified explicitly for the evaluation script.
EVALUATION_NOISE_LEVELS = [0.0, 0.001, 0.01, 0.1]


def run_command(command: list, title: str):
    """
    Executes a command-line process and handles errors.

    Args:
        command (list): The command and its arguments as a list of strings.
        title (str): A descriptive title for the process being run.
    """
    print("\n" + "="*80)
    print(f" STARTING: {title}")
    print(f"   Executing: {' '.join(command)}")
    print("="*80)

    try:
        # Execute the command. check=True will raise an exception on a non-zero exit code.
        subprocess.run(command, check=True, text=True)
        print("\n" + "-"*80)
        print(f" SUCCESS: {title} completed.")
        print("-"*80)
    except FileNotFoundError:
        print(f" ERROR: The script '{command[1]}' was not found in the current directory.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f" ERROR: The process for '{title}' failed with return code {e.returncode}.")
        print("   Please check the output above for specific error messages from the script.")
        sys.exit(1)
    except Exception as e:
        print(f" An unexpected error occurred during '{title}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Get the python interpreter executable to ensure scripts run with the same environment.
    python_executable = sys.executable

    # Main experiment loop
    for algo in ALGORITHMS_TO_RUN:
        # --- 1. Training Step ---
        # Train the current algorithm on all predefined noise levels.
        training_command = [
            python_executable,
            "jax_bbc_train.py",
            "--algo", algo,
            "--noise", "all"  # This flag tells the script to train on 0.0, 0.001, 0.01, 0.1
        ]
        run_command(training_command, f"Training for {algo.upper()}")

        # --- 2. Evaluation Step ---
        # Evaluate the newly trained models against all specified noise environments.
        # Convert the list of floats to a list of strings for the command line.
        eval_noise_str_list = [str(n) for n in EVALUATION_NOISE_LEVELS]

        evaluation_command = [
            python_executable,
            "eval_np_bbc.py",
            "--algo", algo,
            "--eval-noise", *eval_noise_str_list # Unpack the list into individual arguments
        ]
        run_command(evaluation_command, f"Evaluation for {algo.upper()}")

    print("\n" + "*"*80)
    print(" All training and evaluation experiments have completed successfully! 🎉")
    print("*"*80)
