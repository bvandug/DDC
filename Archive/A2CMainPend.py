import os
# 🔧 Fix OpenMP crash on macOS with PyTorch + MATLAB
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import torch
from A2CLearningAgent import agent

def updateProgressBar(episode_num, total_episodes, progress_width=25):
    '''
    Displays a text-based loading bar in the terminal.

    Args:
        episode_num (int): Current episode number.
        total (int): Total number of episodes.
        progress_width (int): The length of the loading bar in characters.
    '''
    progress = int(episode_num / total_episodes * progress_width)
    percentage = min((episode_num / total_episodes)*100, 100)
    
    sys.stdout.write("\r[{}{}] {}/{} episodes ({:.1f}%)".format("=" * progress, "-" * (progress_width - progress), episode_num, total_episodes, percentage))
    sys.stdout.flush()


def viewModel(model_file='a2c_model.pth'):
    ''' 
    Displays information about the A2C model.

    Args:
        model_file (str): Name of file where the model is stored.
    '''
    try:
        model = torch.load(model_file, map_location=torch.device('cpu'))
        print(f"Model loaded from {model_file}")
        for name, param in model.items():
            print(f"{name}: {param.shape}")
    except Exception as e:
        print(f"Error loading model: {e}")

def genIntialAngle(delta=np.pi/3):
    '''
    Generates a random initial angle about 0 radians.
    The result lies in the range [-delta, delta) radians.

    Args:
        delta (float): Magnitude of maximum angular displacement.

    Returns:
        float: Angle in radians.
    '''
    return np.random.uniform(-delta, delta)

def train(eng, model, mask, total_episodes=None, count=0):
    '''
    Trains the A2C Agent from scratch.
    
    Args:
        eng (MatlabEngine Object): The instance of Matlab currently running.
        model (str): Name of the Simulink model in use.
        mask (str): Name of the model block of the system in use.
        total_episodes (int, optional): Total number of episodes for training.
        count (int): Starting episode count.
    '''
    # Use the agent's total_episodes if none provided
    if total_episodes is None:
        total_episodes = agent.total_episodes
    
    print(f"Starting training for {total_episodes} episodes...")
    
    for episode in range(1, total_episodes + 1):
        # Generate initial angle for episode
        initial_angle = genIntialAngle()

        # Set parameters for Simulink
        if 'numEpisodes' in eng.get_param(f'{model}', 'ObjectParameters', nargout=1):
            eng.set_param(f'{model}/numEpisodes', 'Value', str(episode), nargout=0)

        # Check if the mask block exists and set the initial angle
        try:
            eng.set_param(f'{model}/{mask}', 'init', str(initial_angle), nargout=0)
        except Exception as e:
            print(f"Error setting initial angle: {e}")
            print(f"Check if '{mask}' block exists in '{model}'")
            return

        # Simulate one episode
        try:
            eng.sim(model)
            current_episode = count + episode
            
            # Update model after each episode
            agent.update()

            # Save model more frequently (every 10 episodes)
            if current_episode % 10 == 0 or current_episode == total_episodes:
                agent.save_model()

            updateProgressBar(current_episode, total_episodes)
        except Exception as e:
            print(f"\nError in simulation: {e}")
            break
    
    print("\nTraining completed!")


def setNoise(eng, model, noise):
    '''
    Sets amount of noise to be supplied to the state variables.

    Args:
        eng (MatlabEngine Object): The instance of Matlab currently running.
        model (str): Name of the Simulink model in use.
        noise (bool): Whether noise should be supplied or not.
    '''
    try:
        if noise:
            random_seed = np.random.randint(1, 100000)
            eng.set_param(f'{model}/Noise', 'Cov', str([0.00001]), nargout=0)
            eng.set_param(f'{model}/Noise', 'seed', str([random_seed]), nargout=0)
            random_seed = np.random.randint(1, 100000)
            eng.set_param(f'{model}/Noise_v', 'Cov', str([0.001]), nargout=0)
            eng.set_param(f'{model}/Noise_v', 'seed', str([random_seed]), nargout=0)
        else:
            eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)
            eng.set_param(f'{model}/Noise_v', 'Cov', str([0]), nargout=0)
    except Exception as e:
        print(f"Error setting noise parameters: {e}")
        print("Check if 'Noise' and 'Noise_v' blocks exist in the model")

def list_simulink_models(eng):
    """
    Lists all .slx files in the current directory.

    Args:
        eng (MatlabEngine Object): The instance of MATLAB currently running.

    Returns:
        list: List of .slx file names without extension
    """
    try:
        # Get current MATLAB directory
        current_dir = eng.pwd()
        print(f"Current MATLAB directory: {current_dir}")
        
        # Get .slx filenames as a list of strings
        files = eng.eval("cellstr(string({dir('*.slx').name}));", nargout=1)

        # Strip .slx extensions
        model_files = [f[:-4] if f.endswith('.slx') else f for f in files]

        print("Available Simulink models:")
        for model in model_files:
            print(f"  - {model}")

        return model_files
    except Exception as e:
        print(f"Error listing Simulink models: {e}")
        return []


def main(trainModel=True, 
         noise=False,
         trainingModel='pendSimQTraining',
         controllerModel='pendSimQController',
         cartPoleSubsystem='Pendulum and Cart',
         stabilisation_precision=0.05):
    '''
    Method to set up MATLAB, Simulink, and handle data aquisition/plotting.

    Args:
        trainModel (bool): Whether the model should be trained or not.
        noise (bool): Whether noise should be supplied or not.
        trainingModel (str): Name of the Simulink model used for training.
        controllerModel (str): Name of the Simulink model used for controlling.
        cartPoleSubsystem (str): Name of the model block of the system in use.
        stabilisation_precision (float): Magnitude of error bounds around 0 rad.
    '''
    print("Setting up MATLAB engine...")
    try:
        eng = matlab.engine.start_matlab()
    except Exception as e:
        print(f"Error starting MATLAB engine: {e}")
        return
    
    # List available Simulink models
    available_models = list_simulink_models(eng)
    
    # Check and update model names if necessary
    if trainingModel not in available_models:
        if len(available_models) > 0:
            # Use the first available model for training
            trainingModel = available_models[0]
            print(f"Using '{trainingModel}' for training")
        else:
            print("No Simulink models found. Please create a Simulink model first.")
            eng.quit()
            return
    
    if controllerModel not in available_models:
        # Use the same model for controller if not available
        controllerModel = trainingModel
        print(f"Using '{controllerModel}' for controller")
    
    start_time = time.time()
    
    # Training model if specified
    if trainModel:
        print(f"Loading training model: {trainingModel}")
        try:
            eng.load_system(trainingModel, nargout=0)
            print("Training A2C model...")
            train(eng, trainingModel, cartPoleSubsystem, agent.total_episodes)
            
            # Verify model was saved
            model_file = 'a2c_model.pth'
            if os.path.exists(model_file):
                print(f"Model successfully saved to {model_file}")
                viewModel()  # Display information about the trained model
            else:
                print(f"Warning: Expected model file {model_file} not found after training")
        except Exception as e:
            print(f"Error during training: {e}")
    
    # Load controller model (may be the same as training model)
    print(f"Loading controller model: {controllerModel}")
    try:
        if controllerModel != trainingModel or not trainModel:
            eng.load_system(controllerModel, nargout=0)
        
        # Setting controller parameters
        initial_angle = genIntialAngle()
        eng.set_param(f'{controllerModel}/{cartPoleSubsystem}', 'init', str(initial_angle), nargout=0)
        setNoise(eng, controllerModel, noise)
        
        print(f"Running controller simulation with initial angle: {initial_angle:.2f} radians")
        eng.eval(f"out = sim('{controllerModel}');", nargout=0)
        
        # Get angles and times
        try:
            angle_2d = eng.eval("out.angle")
            angle_lst = []
            for a in angle_2d:
                angle_lst.append(a[0])

            time_2d = eng.eval("out.time")
            time_lst = []
            for t in time_2d:
                time_lst.append(t[0])
                
            # Plotting acquired data
            plt.figure(figsize=(10, 6))
            plt.plot(time_lst, angle_lst, label=f"Initial angle: {initial_angle:.2f} rad")
            plt.axhline(y=stabilisation_precision, color='k', linestyle='--')
            plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'± {stabilisation_precision:.2f} rad')
            plt.xlabel("Time (s)")
            plt.ylabel("θ (rad)")
            plt.xlim(0, 3)
            plt.ylim(-np.pi/2, np.pi/2)
            plt.legend(loc='upper right', fontsize="11")
            plt.title("A2C Controller Performance")
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e:
            print(f"Error processing simulation results: {e}")
            print("Check if 'angle' and 'time' variables are available in the simulation output")
        
    except Exception as e:
        print(f"Error in controller simulation: {e}")
    
    # Close MATLAB engine
    eng.quit()
    
    duration = time.time() - start_time
    print(f"Simulation complete in {duration:.1f} secs")

if __name__ == '__main__':
    main(trainModel=True, noise=False)