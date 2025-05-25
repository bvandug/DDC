import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import matlab.engine # Import matlab.engine here

# Import your custom environment
from CustomPendulumEnv import CustomPendulumEnv # Ensure this file is in the same directory or Python path

if __name__ == "__main__":
    shared_matlab_engine = None # Initialize to None
    try:
        # --- Start MATLAB Engine ONCE ---
        print("Starting shared MATLAB engine for the training session...")
        shared_matlab_engine = matlab.engine.start_matlab()
        print("Shared MATLAB engine started.")

        # --- IMPORTANT: User Configuration ---
        SIMULINK_MODEL_NAME = "PendCart"
        TOTAL_TIMESTEPS = 20000  # Adjusted for a more meaningful training session
        ALGORITHM = PPO
        POLICY = "MlpPolicy"
        SIM_DT = 0.02
        MAX_EP_STEPS = 250
        # ---

        log_dir = f"./logs_{ALGORITHM.__name__}_{SIMULINK_MODEL_NAME}/"
        model_save_path = f"./models_{ALGORITHM.__name__}_{SIMULINK_MODEL_NAME}/"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_save_path, exist_ok=True)

        # --- Pass the shared engine to the training environment ---
        env_kwargs_train = {
            'model_name': SIMULINK_MODEL_NAME,
            'dt': SIM_DT,
            'max_steps_per_episode': MAX_EP_STEPS,
            'matlab_engine_instance': shared_matlab_engine # Pass the engine
        }
        # Use a lambda to wrap the class with its arguments for make_vec_env
        # This ensures make_vec_env can correctly pass kwargs when it instantiates the env.
        env = make_vec_env(lambda: CustomPendulumEnv(**env_kwargs_train), n_envs=1)


        # --- Create a separate evaluation environment, also using the shared engine ---
        # Important: If EvalCallback creates its own envs and doesn't take 'env_kwargs',
        # this explicit creation is necessary.
        env_kwargs_eval = {
            'model_name': SIMULINK_MODEL_NAME, # Could be a validation variant of the model if needed
            'dt': SIM_DT,
            'max_steps_per_episode': MAX_EP_STEPS,
            'matlab_engine_instance': shared_matlab_engine # Pass the same engine
        }
        # For EvalCallback, it's better to pass a function that creates the env or an already created env.
        # If passing an already created env, ensure it's not the same instance as the training env if parallel execution is a concern (not with n_envs=1).
        # For simplicity with SB3, creating it separately like this is clear.
        eval_env = CustomPendulumEnv(**env_kwargs_eval)


        # Define callbacks
        eval_callback = EvalCallback(eval_env, # Pass the dedicated evaluation environment
                                     best_model_save_path=model_save_path,
                                     log_path=log_dir,
                                     eval_freq=max(5000 // env.num_envs, 1), # Evaluate more frequently
                                     n_eval_episodes=5,
                                     deterministic=True,
                                     render=False)
        
        checkpoint_callback = CheckpointCallback(save_freq=max(10000 // env.num_envs, 1),
                                                 save_path=model_save_path,
                                                 name_prefix=f"rl_model_{SIMULINK_MODEL_NAME}")

        model = ALGORITHM(
            POLICY,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95
        )

        print(f"Starting training with {ALGORITHM.__name__} on '{SIMULINK_MODEL_NAME}' for {TOTAL_TIMESTEPS} timesteps...")
        print(f"Logs will be saved to: {log_dir}")
        print(f"Models will be saved to: {model_save_path}")

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
    
    except Exception as e:
        print(f"An error occurred during the training script: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    finally:
        if 'model' in locals() and model is not None: # Check if model was initialized
            try:
                model.save(os.path.join(model_save_path, f"final_model_{SIMULINK_MODEL_NAME}_{TOTAL_TIMESTEPS}"))
                print("Training finished or interrupted. Final model saved.")
            except Exception as e:
                print(f"Error saving final model: {e}")
        
        # Close environments (they should no longer quit the engine themselves)
        if 'eval_env' in locals() and eval_env is not None:
            try:
                eval_env.close()
                print("Evaluation environment closed.")
            except Exception as e:
                print(f"Error closing evaluation environment: {e}")
        
        if 'env' in locals() and env is not None: # env from make_vec_env
            try:
                env.close() 
                print("Training environment closed.")
            except Exception as e:
                print(f"Error closing training environment: {e}")

        # --- Quit Shared MATLAB Engine ONCE at the very end ---
        if shared_matlab_engine: # Check if it was successfully started
            try:
                print("Quitting shared MATLAB engine...")
                shared_matlab_engine.quit()
                print("Shared MATLAB engine quit.")
            except Exception as e:
                print(f"Error quitting shared MATLAB engine: {e}")
        print("Script finished.")