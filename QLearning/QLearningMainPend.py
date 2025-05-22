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


def viewTable(qtable_file='qtable.npy'):
    ''' 
    Displays the Q-Table as an array.

    Args:
        qtable_file (str): Name of file where the Q-Table is stored.
    '''
    qtable = np.load(qtable_file)
    np.set_printoptions(precision=1, suppress=True)
    print(qtable)

# reset Q table
def reset(num_states=144, num_actions=4, qtable_file='qtable.npy'):
    '''
    Resets the Q-Table to 2D array of zeros of size num_states x num_actions.

    Args:
        num_actions (int): Number of actions available.
        num_states (int): Number of states available.
        qtable_file (str): Name of file where the Q-Table is stored.
    '''
    qtable = np.zeros((num_states, num_actions))
    np.save(qtable_file, qtable)

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

def train(eng, model, mask, total_episodes=1500, count=0):
    '''
    Trains the Q-Learning Agent from scratch to populate an optimal Q-Table.

    Args:
        eng (MatlabEngine Object): The instance of Matlab currently running.
        model (str): Name of the Simulink model in use.
        mask (str): Name of the model block in use.
        total_episodes (int): Total number of episodes to be completed.
        count (int): The current episode number.
    '''
    reset()

    for episode in range(1,total_episodes+1):

        # Setting training model parameters
        initial_angle = genIntialAngle()
        eng.set_param(f'{model}/numEpisodes', 'Value', str(episode), nargout=0)
        eng.set_param(f'{model}/{mask}', 'init', str(initial_angle), nargout=0)
        
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
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise', 'Cov', str([0.00001]), nargout=0)
        eng.set_param(f'{model}/Noise', 'seed', str([random_seed]), nargout=0)
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise_v', 'Cov', str([0.001]), nargout=0)
        eng.set_param(f'{model}/Noise_v', 'seed', str([random_seed]), nargout=0)
    else:
        eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)
        eng.set_param(f'{model}/Noise_v', 'Cov', str([0]), nargout=0)

def main(trainModel = True, 
         noise = False,
         trainingModel = 'pendSimQTraining', 
         controllerModel = 'pendSimQController',
         cartPoleSubsystem = 'Pendulum and Cart',
         stabilisation_precision = 0.05):
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
    global time

    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(trainingModel, nargout=0)
    start_time = time.time()

    # Training model if specified
    if trainModel:
        print("Training model...")
        train(eng, trainingModel, cartPoleSubsystem)
    
    print("Running simulation...")
    eng.load_system(controllerModel, nargout=0)

    # Setting controller parameters
    initial_angle = genIntialAngle()
    eng.set_param(f'{controllerModel}/{cartPoleSubsystem}', 'init', str(initial_angle), nargout=0)
    setNoise(eng, controllerModel, noise)
    eng.eval(f"out = sim('{controllerModel}');", nargout=0)
    
    # Showing Q-Table and starting angle
    print("Final Q-Table:")
    viewTable()
    print(f"Initial Angle: {initial_angle:.2f} radians")

    # Get angles
    angle_2d = eng.eval("out.angle")
    angle_lst = []
    for a in angle_2d:
        angle_lst.append(a[0])

    # Get times
    time_2d = eng.eval("out.time")
    time_lst = []
    for t in time_2d:
        time_lst.append(t[0])

    eng.quit()

    # Plotting acquired data
    plt.plot(time_lst, angle_lst, label = f"{initial_angle:.2f} rad")
    plt.axhline(y=stabilisation_precision, color='k', linestyle='--')
    plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'± {stabilisation_precision:.2f} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("θ (rad)")
    plt.xlim(0,3)
    plt.ylim(-np.pi/2, np.pi/2)
    plt.legend(loc = 'upper right', fontsize = "11")
    plt.show()

    duration = time.time() - start_time
    print(f"Simulation complete in {duration:.1f} secs")

if __name__ == '__main__':
    main(trainModel=True, noise=False)