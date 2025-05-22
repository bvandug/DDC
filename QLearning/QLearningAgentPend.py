import numpy as np

class QLearningAgent:
    '''
    A class that implements a Q-Learning Agent for an inverted pendulum system.
    
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
        forces (list): Possible actions (forces) that can be taken be the controller.
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
        self.qfile = 'qtable.npy'
        self.qtable = np.load(self.qfile)

        self.total_episodes = 1500

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay_val = 0.995
        self.episode_threshold = self.total_episodes//10
        
        self.last_state = None
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

    def reward_function(self, theta, theta_dot, last_action):
        '''
        Calculates a reward to be distributed based on key state variables.
        (adapted from: https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py)

        Args:
            theta (float): Angle error signal.
            theta_dot (float): Angular velocity.
            last_action (int): Index of last action taken.

        Returns:
            float: The reward due.
        '''
        stabilisation_reward = 10 if np.linalg.norm([theta, theta_dot]) < 0.1 else 0
        angle_penalty = theta**2 
        ang_velocity_penalty = 0.1 * theta_dot**2 
        action_penalty = 0.001 * self.forces[last_action]**2

        return -(angle_penalty + ang_velocity_penalty + action_penalty) + stabilisation_reward

    
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


    def get_force(self, theta, theta_dot, num_episodes):
        '''
        Carries out the steps required to get an output: gets the current state,
        caclulates a reward, updates the Q-Table, selects and returns a new action appropriately.

        Args:
            theta (float): Angle error signal.
            theta_dot (float): Angular velocity.
            num_episodes (float): The current episode number.

        Returns
            float: Output signal (force).
        '''
        state = self.get_state(theta, theta_dot)
        if self.last_state is not None:
            reward = self.reward_function(theta, theta_dot, self.last_action)
            self.update(self.last_state, self.last_action, reward, state, num_episodes)
        action = self.select_action(state, num_episodes)
        self.last_state = state
        self.last_action = action
        force = self.forces[action] 

        return force
           
# Instantiate Q-Learning Agent.
agent = QLearningAgent(alpha=0.1, gamma=0.99)     

def controller_call(rad_big, theta_dot, num_episodes):
    '''
    Calls the Q-Learning Agent to compute the control signal.

    Args:
        rad_big (float): Raw angle error signal.
        theta_dot (float): Angular velocity.
        num_episodes (float): The current episode number.

    Returns:
        float: Output signal (force).
    '''
    global agent

    # Normalise the angle (between -π and π)
    theta = (rad_big%(np.sign(rad_big)*2*np.pi))
    if theta >= np.pi:
        theta -= 2 * np.pi

    force = agent.get_force(theta, theta_dot, num_episodes)
    return force