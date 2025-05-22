import numpy as np

class QLearningController:
    '''
    A class that implements a (trained) Q-Learning Controller for a buck converter system.
    
    Attributes:
        qfile (str): Name of file where the Q-Table is stored.
        qtable (numpy.ndarray): The Agent's Q-Table.
        last_action (int): Previous action taken.
        actions (list): Possible actions (duty cycles) that can be taken be the controller.
        fail_state (int): An extremely undesireable state.
        num_actions (int): Number of actions available.
        num_states (int): Number of states available.
    '''

    def __init__(self):
        '''
        Initialises the Q-Learning Controller for use.
        '''
        self.qfile = 'qtable_BC.npy'
        self.qtable = np.load(self.qfile)

        self.last_action = None
       
        self.actions = [0.1, 0.3, 0.5, 0.7, 0.9]

        self.fail_state = -1
        self.num_actions = len(self.actions)
        self.num_states = 24 # 0 - 23

    def get_state(self, voltage):
        '''
        Converts continous parameters into discrete states.

        Args:
            voltage (float): Voltage error signal.

        Returns:
            int: Current state.
        '''
        if (voltage < -100 or voltage > 100):
            return self.fail_state
        
        # Voltages
        if voltage < -60: box = 0
        elif voltage < -40: box = 1
        elif voltage < -30: box = 2
        elif voltage < -23: box = 3
        elif voltage < -16: box = 4
        elif voltage < -13: box = 5
        elif voltage < -10: box = 6
        elif voltage < -8: box = 7
        elif voltage < -5: box = 8
        elif voltage < -3: box = 9
        elif voltage < -1: box = 10
        elif voltage < 0: box = 11
        elif voltage < 1: box = 12
        elif voltage < 3: box = 13
        elif voltage < 5: box = 14
        elif voltage < 8: box = 15
        elif voltage < 10: box = 16
        elif voltage < 13: box = 17
        elif voltage < 16: box = 18
        elif voltage < 23: box = 19
        elif voltage < 30: box = 20
        elif voltage < 40: box = 21
        elif voltage < 60: box = 22
        else: box = 23

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

    def get_output(self, voltage):
        '''
        Carries out the steps required to get an output: gets the current state
        and then selects and returns an action accordingly.

        Args:
            voltage (float): Voltage error signal.

        Returns
            float: Output signal (duty cycle).
        '''
        state = self.get_state(voltage)
        action = self.select_action(state)
        self.last_state = state
        self.last_action = action
        duty_cycle = self.actions[action]
        return duty_cycle
        
# Instantiate Q-Learning Controller
controller = QLearningController()   

def controller_call(voltage):
    '''
    Calls the Q-Learning Controller to compute the control signal.

    Args:
        voltage (float): Voltage error signal.

    Returns:
        float: Output signal (duty cycle).
    '''
    global controller

    output = controller.get_output(voltage)
    return output