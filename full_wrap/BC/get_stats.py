import os
import gymnasium as gym
from stable_baselines3 import A2C
import torch
from gymnasium.spaces import Box # Import Box for type hinting if needed

# Import the custom environment and wrappers used during training
from BCSimTestEnv import BCSimulinkEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class SB3ModelInspector:
    """
    A class to inspect a trained Stable-Baselines3 A2C model.

    This class allows you to load an A2C model from a .zip file and
    then retrieve information about the total timesteps it was trained
    with, as well as its policy network parameters and all
    algorithm-specific hyperparameters.
    """

    def __init__(self, model_path: str, env_instance: gym.Env = None):
        """
        Initializes the SB3ModelInspector by loading the A2C model.

        Args:
            model_path (str): The file path to the saved A2C model (.zip).
            env_instance (gym.Env): An instance of the Gymnasium environment
                                    (including any wrappers like VecNormalize)
                                    that the model was trained on.
                                    This is crucial for matching observation/action spaces.
        """
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please ensure the model file exists at the specified path.")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if env_instance is None:
            raise ValueError("An environment instance (env_instance) must be provided to load the model.")

        self.model_path = model_path
        self.env_instance = env_instance
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the A2C model from the specified path using the provided environment instance.
        """
        try:
            print("Loading model with provided environment instance...")
            self.model = A2C.load(self.model_path, env=self.env_instance)
            print(f"Successfully loaded A2C model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {e}")
            print(f"Observed error: {e}") # Print the specific error for debugging
            self.model = None

    def get_total_timesteps(self) -> int:
        """
        Returns the total number of timesteps the model was trained for.

        Returns:
            int: The total number of timesteps. Returns 0 if the model is not loaded.
        """
        if self.model:
            return self.model.num_timesteps
        return 0

    def get_policy_parameters(self):
        """
        Prints the shapes of the policy network's trainable parameters.
        """
        if self.model:
            print("\n--- Policy Network Parameters (Shapes) ---")
            # Iterate through named parameters of the policy network
            for name, param in self.model.policy.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: {param.shape}")
        else:
            print("Model not loaded. Cannot retrieve policy parameters.")

    def get_algorithm_hyperparameters(self):
        """
        Prints all algorithm-specific and policy-related hyperparameters of the A2C model.
        """
        if self.model:
            print("\n--- A2C Algorithm Hyperparameters ---")
            print(f"  Learning Rate: {self.model.lr_schedule(1.0):.6f}")
            print(f"  Gamma (Discount Factor): {self.model.gamma}")
            print(f"  GAE Lambda: {self.model.gae_lambda}")
            print(f"  Entropy Coefficient: {self.model.ent_coef}")
            print(f"  Value Function Coefficient: {self.model.vf_coef}")
            print(f"  Max Grad Norm: {self.model.max_grad_norm}")
            print(f"  Number of Steps (n_steps): {self.model.n_steps}")
            # Use getattr for safety with older model versions
            print(f"  Normalize Advantage: {getattr(self.model, 'normalize_advantage', 'N/A')}")

            print("\n  --- Policy & Network Architecture ---")
            # policy.activation_fn stores the class, .__name__ gets the string (e.g., 'ReLU')
            print(f"  Activation Function: {self.model.policy.activation_fn.__name__}")
            # The net_arch attribute is part of the policy's arguments
            net_arch = getattr(self.model.policy, 'net_arch', 'N/A')
            print(f"  Network Architecture (net_arch): {net_arch}")

            print("\n  --- Optimizer ---")
            if self.model.optimizer and self.model.optimizer.param_groups:
                # The use_rms_prop is inferred by the optimizer's class name
                print(f"  Optimizer Class: {self.model.optimizer.__class__.__name__}")
                print(f"  RMSprop Epsilon: {self.model.optimizer.param_groups[0].get('eps', 'N/A')}")
                print(f"  RMSprop Alpha: {self.model.optimizer.param_groups[0].get('alpha', 'N/A')}")
            else:
                print("  Optimizer parameters not accessible.")

            print("\n  --- State-Dependent Exploration (SDE) ---")
            print(f"  Use SDE: {self.model.use_sde}")
            if self.model.use_sde:
                print(f"  SDE Sample Frequency: {self.model.sde_sample_freq}")
        else:
            print("Model not loaded. Cannot retrieve algorithm hyperparameters.")

# --- Example Usage ---
if __name__ == "__main__":
    # Define the path to your model file and normalization stats
    model_file = "model_400000.zip"
    vec_normalize_stats_file = "vec_normalize_400000.pkl"

    # --- Environment Instantiation with Parameters from train_bc.py ---
    # These values are taken directly from your train_bc.py script
    EPISODE_TIME = 0.03
    GRACE_PERIOD = 50

    print("Setting up the environment for model loading...")

    # Create the base custom environment function
    env_fn = lambda: BCSimulinkEnv(
        model_name="bcSim",
        frame_skip=10,
        enable_plotting=False,
        grace_period_steps=GRACE_PERIOD,
        max_episode_time=EPISODE_TIME,
    )

    # Wrap the environment in DummyVecEnv
    env = DummyVecEnv([env_fn])

    # Wrap the environment in VecNormalize and load the saved stats
    if os.path.exists(vec_normalize_stats_file):
        try:
            # First, create the VecNormalize instance with the correct parameters
            env = VecNormalize(env,
                               norm_obs=True,
                               norm_reward=False,
                               clip_obs=10.0)
            # Then, load the saved statistics into it
            env = VecNormalize.load(vec_normalize_stats_file, env)
            print(f"Successfully loaded VecNormalize stats from {vec_normalize_stats_file}")
        except Exception as e:
            print(f"Error loading VecNormalize stats: {e}")
            print("Proceeding without normalization, which may cause issues.")
    else:
        print(f"Warning: VecNormalize stats file '{vec_normalize_stats_file}' not found.")
        print("Creating VecNormalize without loading stats. This might lead to observation space mismatch.")
        env = VecNormalize(env,
                           norm_obs=True,
                           norm_reward=False,
                           clip_obs=10.0)


    print(f"Observation space of environment used for loading: {env.observation_space}")
    print(f"Action space of environment used for loading: {env.action_space}")

    # Now, pass the fully wrapped environment instance to the inspector
    print(f"\nAttempting to inspect model: {model_file}")

    try:
        inspector = SB3ModelInspector(model_file, env_instance=env)

        if inspector.model: # Check if the model was successfully loaded
            timesteps = inspector.get_total_timesteps()
            print(f"\nTotal timesteps the model was trained for: {timesteps}")

            inspector.get_policy_parameters()

            # This will now print the comprehensive list of hyperparameters
            inspector.get_algorithm_hyperparameters()
        else:
            print("\nModel could not be loaded. Please ensure the path is correct and the environment instance matches.")
            
    except FileNotFoundError as e:
        print(f"\nCaught expected error: {e}")
    except ImportError as e:
        print(f"\nCaught an ImportError: {e}")
        print("This often indicates a problem with your library installations (e.g., NumPy, PyTorch).")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Ensure the environment (and MATLAB engine) is closed
        if 'env' in locals() and env:
            env.close()
            print("\nEnvironment closed.")