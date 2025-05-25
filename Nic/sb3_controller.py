import numpy as np
from stable_baselines3 import PPO # Or SAC, TD3, etc. - use the same algo you trained with
import os

# --- Global variable to store the loaded model ---
# This ensures the model is loaded only once when Simulink starts,
# not every time the controller_call function is invoked.
_sb3_model = None
_model_loaded_successfully = False

# --- User Configuration ---
# IMPORTANT: Path to your TRAINED SB3 model file (e.g., from training with train_custom_pendulum.py)
# This path should be accessible from where MATLAB/Simulink is running.
# It's often best to use an absolute path or ensure the model is in MATLAB's path.
MODEL_PATH = "models_PPO_PendCart/final_model_PendCart_200000.zip" # EXAMPLE PATH - CHANGE THIS
# Or, for the final model:
# MODEL_PATH = "models_PPO_YourActualModelName/final_model_YourActualModelName_200000.zip"

# Specify the algorithm class used for training (PPO, SAC, TD3, etc.)
ALGORITHM_CLASS = PPO
# ---

def load_sb3_model_once():
    """Loads the SB3 model if it hasn't been loaded yet."""
    global _sb3_model
    global _model_loaded_successfully
    if _sb3_model is None and not _model_loaded_successfully:
        print(f"Attempting to load SB3 model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            print("Please ensure MODEL_PATH in sb3_simulink_controller.py is correct.")
            _model_loaded_successfully = False # Mark as failed
            return
        try:
            _sb3_model = ALGORITHM_CLASS.load(MODEL_PATH)
            _model_loaded_successfully = True
            print("SB3 model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load SB3 model from {MODEL_PATH}. Exception: {e}")
            _model_loaded_successfully = False # Mark as failed
            # _sb3_model will remain None
    elif _sb3_model is None and not _model_loaded_successfully:
        # Model loading was attempted before and failed, don't try again every call.
        # This state indicates a setup error (e.g. wrong path)
        pass


def controller_call(theta, theta_v, sim_time):
    """
    This function is called by a MATLAB Function block in Simulink.
    It uses a pre-trained Stable Baselines3 agent to determine the control action.

    Args:
        theta (float): Current pendulum angle (radians).
        theta_v (float): Current pendulum angular velocity (radians/sec).
        sim_time (float): Current simulation time (seconds). Used here mainly for the initial model load.

    Returns:
        float: The force to be applied to the cart.
    """
    global _sb3_model
    global _model_loaded_successfully

    # Load the model at the very first call (e.g., when sim_time is 0 or near 0)
    # or if it failed to load previously and we want to give it one shot per simulation run.
    if sim_time == 0.0: # Reset loading attempt for a new simulation run
        _sb3_model = None
        _model_loaded_successfully = False
        print("controller_call: sim_time is 0.0, resetting model load state.")

    if _sb3_model is None and not _model_loaded_successfully:
        load_sb3_model_once()

    if _sb3_model is None:
        # If model loading failed, return a safe/neutral action (e.g., 0 force)
        # and print an error message if it hasn't been printed excessively.
        if not hasattr(controller_call, "load_error_printed"):
            print("CRITICAL: SB3 model is not loaded in controller_call. Returning 0 force.")
            controller_call.load_error_printed = True # Print only once per simulation run
        return 0.0
    else:
        # Ensure load_error_printed attribute exists and reset if model is now loaded
        if hasattr(controller_call, "load_error_printed"):
            del controller_call.load_error_printed


    # Prepare the observation for the SB3 model
    # The observation must match the observation_space of the trained model.
    # Our CustomPendulumEnv uses [angle, angular_velocity].
    observation = np.array([theta, theta_v], dtype=np.float32)

    # Get action from the loaded SB3 model
    # deterministic=True means we take the action with the highest probability (no exploration)
    action, _states = _sb3_model.predict(observation, deterministic=True)

    # The action is usually a NumPy array, extract the scalar value.
    force = float(action[0])

    # Optional: Clip the force if necessary, though the agent should learn limits.
    # max_force_simulink = 10.0 # Should match your system's actual limits
    # force = np.clip(force, -max_force_simulink, max_force_simulink)

    return force

# Example of how you might test this script lightly from Python (won't run Simulink)
if __name__ == '__main__':
    print("sb3_simulink_controller.py: Main execution (for testing purposes).")
    # This won't actually call from Simulink but can help catch Python errors.

    # --- IMPORTANT: Manually set MODEL_PATH for this test if it's not already correct ---
    # MODEL_PATH = "path/to/your/best_model.zip" # Ensure this is a valid path to a trained model
    # ALGORITHM_CLASS = PPO # Ensure this matches the model
    # ---

    if os.path.exists(MODEL_PATH):
        print(f"Test: Attempting to load model: {MODEL_PATH}")
        load_sb3_model_once()
        if _sb3_model:
            print("Test: Model loaded.")
            # Simulate a few calls
            test_obs_1 = controller_call(0.1, 0.0, 0.0) # First call, sim_time = 0
            print(f"Test call 1 (theta=0.1, theta_v=0.0, time=0.0) -> Force: {test_obs_1}")
            test_obs_2 = controller_call(0.0, 0.2, 0.02)
            print(f"Test call 2 (theta=0.0, theta_v=0.2, time=0.02) -> Force: {test_obs_2}")
            test_obs_3 = controller_call(-0.1, -0.1, 0.04)
            print(f"Test call 3 (theta=-0.1, theta_v=-0.1, time=0.04) -> Force: {test_obs_3}")
        else:
            print(f"Test: Model could not be loaded from {MODEL_PATH}. Cannot run controller calls.")
    else:
        print(f"Test: MODEL_PATH '{MODEL_PATH}' does not exist. Cannot run controller calls.")