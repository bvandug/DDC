import numpy as np

class QLearningAgent:
    '''
    A class that implements a Q-Learning Agent for a buck converter system.
    
    Attributes:
        qfile (str): Name of file where the Q-Table is stored.
        qtable (numpy.ndarray): The Agent's Q-Table.
        total_episodes (int): Total number of episodes to be completed.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        min_epsilon (float): Minimum exploration rate.
        epsilon_decay_val (float): Parameter which quantifies the rate at which epsilon decays.
        episode_threshold (int): Episode at which epsilon decay starts.
        last_action (int): Previous action taken.
        last_state (int): Previous state occupied.
        actions (list): Possible actions (duty cycles) that can be taken be the controller.
        fail_state (int): An extremely undesireable state.
        num_actions (int): Number of actions available.
        num_states (int): Number of states available.
    '''

    def __init__(self, alpha, gamma):
        '''
        Initialises the Q-Learning Agent for training.

        Args:
            alpha (float): Learning rate.
            gamma (float): Discount factor.
        '''
        self.qfile = 'qtable_BC.npy'
        self.qtable = np.load(self.qfile)

        self.total_episodes = 1250

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay_val = 0.995
        self.episode_threshold = self.total_episodes//10

        self.last_state = None
        self.last_action = None
        self.actions = [0.1, 0.3, 0.5, 0.7, 0.9]

        self.fail_state = -1
        self.num_actions = len(self.actions)
        self.num_states =  24 # 0 - 23
 
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
            

    def select_action(self, state, num_episodes):
        '''
        Selects the next action using an epsilon greedy strategy with epsilon decay.

        Args:
            state (int): The current state.
            num_episodes (int): The current episode number.

        Returns:
            int: Index of action taken.
        '''
        if self.epsilon > self.min_epsilon and num_episodes > self.episode_threshold:
            self.decay_epsilon(num_episodes)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.qtable[state, :]) 
        
    def decay_epsilon(self, num_episodes):
        '''
        The value of epsilon decays after an epsiode threshold is reached.

        Args: 
            num_episodes (int): The current episode number.
        '''
        self.epsilon = np.power(self.epsilon_decay_val, num_episodes - self.episode_threshold)

    def reward_function(self, voltage, ref_voltage):
        '''
        Calculates a reward to be distributed based on key state variables.
        Two reward functions are provided, the uncommented one is the best-performing.

        Args:
            voltage (float): Voltage error signal.
            ref_voltage (float): Desired/reference voltage. 

        Returns:
            float: The reward due.
        '''
        # First reward function
        return 1/(1 + (voltage)**2)
        
        '''
        # Second (worse-performing) reward function
        v_out = abs(voltage - ref_voltage)
        if (v_out >= 3/4 * ref_voltage) and (v_out <= 5/4 * ref_voltage):
            return (abs(ref_voltage/4) - abs(voltage))**2
        return 0
        '''
    
    def update(self, state, action, reward, next_state, num_episodes):
        '''
        Implementation of the Q-Learning equation, which also saves the
        final Q-Table at the end of the last episode.

        Args:
            state (int): The current state.
            action (int): The current action.
            reward (float): The reward due based on the reward function.
            next_state (int): The next state.
            num_episodes (int): The current episode number.
        '''
        q_old = self.qtable[state, action]
        q_new = reward + self.gamma * np.max(self.qtable[next_state])
        self.qtable[state, action] = q_old + self.alpha * (q_new - q_old)

        # Save updated Q-Table
        if num_episodes == self.total_episodes:
            np.save(self.qfile, self.qtable)

    def get_output(self, voltage, ref_voltage, num_episodes):
        '''
        Carries out the steps required to get an output: gets the current state,
        caclulates a reward, updates the Q-Table, selects and returns a new action appropriately.

        Args:
            voltage (float): Voltage error signal.
            ref_voltage (float): Desired/reference voltage. 
            num_episodes (float): The current episode number.

        Returns
            float: Output signal (duty cycle).
        '''
        state = self.get_state(voltage)
        if self.last_state is not None:
            reward = self.reward_function(voltage, ref_voltage)
            self.update(self.last_state, self.last_action, reward, state, num_episodes)
        action = self.select_action(state, num_episodes)
        self.last_state = state
        self.last_action = action
        duty_cycle = self.actions[action]

        return duty_cycle    
        
# Instantiate QLearningAgent.
agent = QLearningAgent(alpha=0.01, gamma=0.995)     

def controller_call(voltage, ref_voltage, num_episodes):
    '''
    Calls the Q-Learning Agent to compute the control signal.

    Args:
        voltage (float): Voltage error signal.
        ref_voltage (float): Desired/reference voltage. 
        num_episodes (float): The current episode number.

    Returns:
        float: Output signal (duty cycle).
    '''
    global agent
    output = agent.get_output(voltage, ref_voltage, num_episodes)
    return output