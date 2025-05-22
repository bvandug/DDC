import numpy as np

class QLearningController:
    '''
    A class that implements a (trained) Q-Learning Controller for an inverted pendulum system.
    
    Attributes:
        qfile (str): Name of file where the Q-Table is stored.
        qtable (numpy.ndarray): The Agent's Q-Table.
        last_action (int): Previous action taken.
        forces (list): Possible actions (forces) that can be taken be the controller.
        fail_state (int): An extremely undesireable state.
        num_actions (int): Number of actions available.
        num_states (int): Number of states available.
    '''

    def __init__(self):
        '''
        Initialises the Q-Learning Controller for use.
        '''
        self.qfile = 'qtable.npy'
        self.qtable = np.load(self.qfile)

        self.last_action = None

        self.forces = [10.0, 30.0, -10.0, -30.0]

        self.fail_state = -1
        self.num_actions = len(self.forces)
        self.num_states = 144 # 0 - 143
        

    def get_state(self, theta, theta_dot):
        '''
        Converts continous parameters into discrete states.
        (adapted from: https://pages.cs.wisc.edu/~finton/qcontroller.html)
        
        Args:
            theta (float): Angle error signal.
            theta_dot (float): Angular velocity.

        Returns:
            int: Current state.
        '''
        # Convert to degrees
        theta = np.rad2deg(theta)
        theta_dot = np.rad2deg(theta_dot)

        # Failure state
        if theta < -60 or theta > 60:
            return self.fail_state
        
        # Angles
        if (theta < -51): box = 0
        elif(theta < -46): box = 1
        elif (theta < -41): box = 2
        elif (theta < -36): box = 3
        elif (theta < -31): box = 4
        elif(theta < -26): box = 5
        elif (theta < -21): box = 6
        elif(theta < -16): box = 7
        elif (theta < -11): box = 8
        elif (theta < -6): box = 9
        elif (theta < -1): box = 10
        elif (theta < 0): box = 11
        elif (theta < 1): box = 12
        elif (theta < 6): box = 13
        elif (theta < 11): box = 14
        elif(theta < 16): box = 15
        elif (theta < 21): box = 16
        elif (theta < 26): box = 17
        elif(theta < 31): box = 18
        elif (theta < 36): box = 19
        elif (theta < 41): box = 20
        elif (theta < 46): box = 21
        elif(theta < 51): box = 22
        else: box = 23

        # Angular velocities
        if (theta_dot < -50): pass
        elif (theta_dot < -25): box += 24
        elif (theta_dot < 0): box += 48
        elif (theta_dot < 25): box += 72
        elif (theta_dot < 50):  box += 96
        else: box += 120

        return box

    def select_action(self, state):
        '''
        Selects the next action using an exclusively greedy strategy.

        Args:
            state (int): The current state.

        Returns:
            int: Index of action taken.
        '''
        return np.argmax(self.qtable[state, :]) 

    def get_force(self, theta, theta_dot):
        '''
        Carries out the steps required to get an output: gets the current state
        and then selects and returns an action accordingly.

        Args:
            theta (float): Angle error signal.
            theta_dot (float): Angular velocity.

        Returns
            float: Output signal (force).
        '''
        state = self.get_state(theta, theta_dot)
        action = self.select_action(state)
        self.last_state = state
        self.last_action = action
        force = self.forces[action]

        return force
        
# Instantiate Q-Learning Controller
controller = QLearningController()   

def controller_call(rad_big, theta_dot):
    '''
    Calls the Q-Learning Controller to compute the control signal.

    Args:
        theta (float): Angle error signal.
        theta_dot (float): Angular velocity.

    Returns:
        float: Output signal (force).
    '''
    global controller

    # Normalise the angle (between -π and π)
    theta = (rad_big%(np.sign(rad_big)*2*np.pi))
    if theta >= np.pi:
        theta -= 2 * np.pi
        
    force = controller.get_force(theta, theta_dot)
    return force