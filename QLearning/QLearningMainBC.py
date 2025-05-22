import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

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

def viewTable(qtable_file='qtable_BC.npy'):
    ''' 
    Displays the Q-Table as an array.

    Args:
        qtable_file (str): Name of file where the Q-Table is stored.
    '''
    qtable = np.load(qtable_file)
    np.set_printoptions(precision=1, suppress=True)
    print(qtable)

# reset Q table
def reset(num_states=24, num_actions=5, qtable_file='qtable_BC.npy'):
    '''
    Resets the Q-Table to 2D array of zeros of size num_states x num_actions.

    Args:
        num_actions (int): Number of actions available.
        num_states (int): Number of states available.
        qtable_file (str): Name of file where the Q-Table is stored.
    '''
    qtable = np.zeros((num_states, num_actions))
    np.save(qtable_file, qtable)

def select_reference_voltage(v_ref, variation = 0.05):
    '''
    Generates a random desired voltage about the reference voltage.
    The result lies in the range [(1 - variation) * v_ref, (1 + variation) * v_ref) radians.

    Args:
        v_ref (float): Desired/reference voltage. 
        variation (float): Percentage applied to the reference voltage. 

    Returns:
        float: Variated reference voltage.
    '''
    variation_range = v_ref * variation
    return v_ref + np.random.uniform(-variation_range, variation_range)

def train(eng, model, source_voltage, v_ref, total_episodes=1250, count=0):
    '''
    Trains the Q-Learning Agent from scratch to populate an optimal Q-Table.

    Args:
        eng (MatlabEngine Object): The instance of Matlab currently running.
        model (str): Name of the Simulink model in use.
        source_voltage (float): Input voltage.
        v_ref (float): Desired/reference voltage. 
        total_episodes (int): Total number of episodes to be completed.
        count (int): The current episode number.
    '''
    reset()
    for episode in range(1, total_episodes+1):
        
        # Setting training model parameters
        eng.set_param(f'{model}/numEpisodes', 'Value', str(episode), nargout=0)
        eng.set_param(f'{model}/finalVoltage', 'Value', str(select_reference_voltage(v_ref)), nargout=0)
        eng.set_param(f'{model}/input_voltage', 'Amplitude', str(source_voltage), nargout=0)

        # Simulating episodes
        eng.sim(model)
        count += 1
        updateProgressBar(count, total_episodes)
    
    print()

def setNoise(eng, model, noise):
    '''
    Sets amount of noise to be supplied to the state variables.

    Args:
        eng (MatlabEngine Object): The instance of Matlab currently running.
        model (str): Name of the Simulink model in use.
        noise (bool): Whether noise should be supplied or not.
    '''
    if noise:
        eng.set_param(f'{model}/Noise', 'Cov', str([0.0000004]), nargout=0)
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise', 'seed', str([random_seed]), nargout=0)
    else:
        eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)

def main(trainModel = True, 
         noise = False,
         trainingModel = 'bcSimQTraining', 
         controllerModel = 'bcSimQController',
         stabilisation_precision = 0.5,
         source_voltage = 48.0,
         desired_voltage = 30.0):
    '''
    Method to set up MATLAB, Simulink, and handle data aquisition/plotting.

    Args:
        trainModel (bool): Whether the model should be trained or not.
        noise (bool): Whether noise should be supplied or not.
        trainingModel (str): Name of the Simulink model used for training.
        controllerModel (str): Name of the Simulink model used for controlling.
        stabilisation_precision (float): Magnitude of error bounds around the reference voltage.
        source_voltage (float): Input voltage.
        desired_voltage (float): Desired/reference voltage.
    '''
    global time

    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(trainingModel, nargout=0)
    start_time = time.time()

    # Training model if specified
    if trainModel:
        print("Training model...")
        train(eng, trainingModel, source_voltage, desired_voltage)
    
    print("Running simulation...")
    eng.load_system(controllerModel, nargout=0)

    # Setting controller model parameters
    setNoise(eng, controllerModel, noise)
    eng.set_param(f'{controllerModel}/finalVoltage', 'Value', str(desired_voltage), nargout=0)
    eng.set_param(f'{controllerModel}/input_voltage', 'Amplitude', str(source_voltage), nargout=0)
    eng.eval(f"out = sim('{controllerModel}');", nargout=0)
    
    # Showing Q-Table
    print("Final Q-Table:")
    viewTable()

    # Get voltages
    voltage_2d = eng.eval("voltage")
    voltage_lst = []
    for v in voltage_2d:
        voltage_lst.append(v[0])

    # Get times
    time_2d = eng.eval("time")
    time_lst = []
    for t in time_2d:
        time_lst.append(t[0])

    eng.quit()    
    
    # Plotting acquired data
    plt.plot(time_lst, voltage_lst, label = f"Ref: {desired_voltage} V")
    plt.axhline(y=desired_voltage + stabilisation_precision, color='k', linestyle='--')
    plt.axhline(y=desired_voltage - stabilisation_precision, color='k', linestyle='--', label=f'± {stabilisation_precision:.1f} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.xlim(0,0.1)
    plt.legend(loc = 'upper right')
    plt.show()

    duration = time.time() - start_time
    print(f"Simulation complete in {duration:.1f} secs")
    
if __name__ == '__main__':
    main(trainModel=True, noise=False)