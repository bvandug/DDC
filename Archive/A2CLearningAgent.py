import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os

class ActorCriticNetwork(nn.Module):
    '''
    Neural network for the A2C algorithm with both actor and critic heads.
    
    Attributes:
        fc1 (nn.Linear): First shared fully connected layer
        actor_fc (nn.Linear): Actor's hidden layer
        actor_out (nn.Linear): Actor's output layer (policy)
        critic_fc (nn.Linear): Critic's hidden layer
        critic_out (nn.Linear): Critic's output layer (value function)
    '''
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 128)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(128, 64)
        self.actor_out = nn.Linear(64, action_dim)
        
        # Critic head (value)
        self.critic_fc = nn.Linear(128, 64)
        self.critic_out = nn.Linear(64, 1)
        
    def forward(self, x):
        '''
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            tuple: (action_probs, state_value)
        '''
        x = F.relu(self.fc1(x))
        
        # Actor: outputs action probabilities
        actor = F.relu(self.actor_fc(x))
        action_probs = F.softmax(self.actor_out(actor), dim=-1)
        
        # Critic: outputs state value
        critic = F.relu(self.critic_fc(x))
        state_value = self.critic_out(critic)
        
        return action_probs, state_value


class A2CLearningAgent:
    '''
    A class that implements an A2C Agent for an inverted pendulum system.
    
    Attributes:
        model_file (str): Name of file where the model is stored.
        network (ActorCriticNetwork): The Actor-Critic neural network.
        optimizer (torch.optim): Optimizer for training the network.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        forces (list): Possible actions (forces) that can be taken by the controller.
        total_episodes (int): Total number of episodes to be completed.
        gamma (float): Discount factor
        saved_actions (list): List to store actions taken during an episode.
        saved_states (list): List to store states encountered during an episode.
        saved_rewards (list): List to store rewards received during an episode.
        saved_values (list): List to store state values predicted during an episode.
        saved_log_probs (list): List to store log probabilities of actions taken.
        device (torch.device): Device on which to run the neural network.
    '''

    def __init__(self, gamma=0.99, learning_rate=0.001):
        '''
        Initializes the A2C Agent for training.

        Args:
            gamma (float): Discount factor.
            learning_rate (float): Learning rate for the optimizer.
        '''
        self.model_file = 'a2c_model.pth'
        self.state_dim = 2  # theta and theta_dot
        self.action_dim = 7  # Expanded action space for better control granularity
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Try to load existing model with better error handling
        try:
            if os.path.exists(self.model_file):
                self.network.load_state_dict(torch.load(self.model_file, map_location=self.device))
                self.network.eval()
                print(f"Loaded model from {self.model_file}")
            else:
                print(f"No existing model found at {self.model_file}. Starting from scratch.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch.")
            
        self.total_episodes = 3000  # Increased from 1500 for better training
        self.gamma = gamma
        
        # Lists to store episode data
        self.reset_episode_memory()
        
        # Expanded action space for finer control
        self.forces = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]

    def reset_episode_memory(self):
        '''
        Resets the memory for a new episode.
        '''
        self.saved_actions = []
        self.saved_states = []
        self.saved_rewards = []
        self.saved_values = []
        self.saved_log_probs = []

    def select_action(self, state):
        '''
        Selects an action using the current policy.

        Args:
            state (numpy.ndarray): The current state [theta, theta_dot].

        Returns:
            int: Index of the selected action.
        '''
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Forward pass through the network
        probs, value = self.network(state_tensor)
        
        # Create a distribution from the probabilities
        m = Categorical(probs)
        
        # Sample an action
        action = m.sample()
        
        # Save data for training
        self.saved_actions.append(action.item())
        self.saved_states.append(state)
        self.saved_log_probs.append(m.log_prob(action))
        self.saved_values.append(value)
        
        return action.item()

    def reward_function(self, theta, theta_dot, action_idx):
        '''
        Improved reward function with better stability incentives and smoother gradient.
        '''
        # Primary reward for being upright (highest at theta=0, smoothly decreases as theta increases)
        upright_reward = 2.0 * np.cos(theta)  # Max 2.0 when theta=0
        
        # Strong bonus for being very close to upright position
        stability_bonus = 3.0 * np.exp(-12.0 * theta**2)  # Gaussian peak at theta=0
        
        # Angular velocity penalty (stronger for high velocities, quadratic)
        velocity_penalty = 0.2 * theta_dot**2
        
        # Penalty for using large forces (energy efficiency)
        force_magnitude = abs(self.forces[action_idx])
        action_penalty = 0.005 * force_magnitude
        
        # Special bonus for perfect stability (near zero angle and velocity)
        perfect_bonus = 2.0 if (abs(theta) < 0.05 and abs(theta_dot) < 0.1) else 0.0
        
        # Combined reward (higher is better)
        reward = upright_reward + stability_bonus - velocity_penalty - action_penalty + perfect_bonus
        
        # Extra penalty for falling over (theta near ±π)
        if abs(theta) > 0.8 * np.pi:
            reward -= 5.0
            
        return reward

    def update(self, final_value=0):
        '''
        Updates the A2C network using collected trajectory data.
        
        Args:
            final_value (float): The estimated value of the final state.
        '''
        if len(self.saved_rewards) == 0:
            return
            
        # Convert lists to tensors
        rewards = torch.FloatTensor(self.saved_rewards).to(self.device)
        values = torch.cat(self.saved_values).to(self.device)
        log_probs = torch.stack(self.saved_log_probs).to(self.device)
        
        # Calculate returns
        returns = []
        R = final_value
        for r in rewards.flip(dims=[0]):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Calculate advantages
        advantages = returns - values
        
        # Calculate losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        
        # Combined loss with entropy bonus for exploration
        entropy = -(log_probs * log_probs.exp()).mean()  # Add entropy term to encourage exploration
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)  # Gradient clipping for stability
        self.optimizer.step()
        
        # Reset episode memory
        self.reset_episode_memory()
        
    def save_model(self):
        '''
        Saves the model to a file.
        '''
        torch.save(self.network.state_dict(), self.model_file)
        print(f"Model saved to {self.model_file}")

    def get_force(self, theta, theta_dot, num_episodes=None, train=True):
        '''
        Computes the control signal and collects experience if in training mode.

        Args:
            theta (float): Angle error signal.
            theta_dot (float): Angular velocity.
            num_episodes (int, optional): The current episode number (for logging).
            train (bool): Whether to train the model or just use it.

        Returns:
            float: Output signal (force).
        '''
        # Create state vector
        state = np.array([theta, theta_dot])
        
        # Select action
        action_idx = self.select_action(state)
        
        # Get force from selected action
        force = self.forces[action_idx]
        
        # Add damping for stability
        if train:
            # Regular force during training
            force_with_damping = force
        else:
            # Add damping during testing for extra stability
            damping = -0.5 * theta_dot
            force_with_damping = force + damping
        
        if train:
            # Calculate reward
            reward = self.reward_function(theta, theta_dot, action_idx)
            
            # Store reward
            self.saved_rewards.append(reward)
        
        return force_with_damping

# Instantiate A2C Agent
agent = A2CLearningAgent(gamma=0.99, learning_rate=0.001)

def controller_call(rad_big, theta_dot, num_episodes=None, train=True):
    '''
    Calls the A2C Agent to compute the control signal.

    Args:
        rad_big (float): Raw angle error signal.
        theta_dot (float): Angular velocity.
        num_episodes (int, optional): The current episode number.
        train (bool): Whether to train the agent or just use it.

    Returns:
        float: Output signal (force).
    '''
    global agent

    # Normalize the angle (between -π and π)
    theta = (rad_big % (2*np.pi))
    if theta > np.pi:
        theta -= 2 * np.pi

    force = agent.get_force(theta, theta_dot, num_episodes, train)
    
    # Safety limiter
    force = np.clip(force, -50.0, 50.0)
    
    return force