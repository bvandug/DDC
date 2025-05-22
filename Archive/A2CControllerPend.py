import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class A2CController:
    '''
    A class that implements a (trained) A2C Controller for an inverted pendulum system.
    
    Attributes:
        model_file (str): Name of file where the model is stored.
        network (ActorCriticNetwork): The Actor-Critic neural network.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        forces (list): Possible actions (forces) that can be taken by the controller.
        device (torch.device): Device on which to run the neural network.
    '''

    def __init__(self):
        '''
        Initializes the A2C Controller for use.
        '''
        self.model_file = 'a2c_model.pth'
        self.state_dim = 2  # theta and theta_dot
        self.action_dim = 7  # Expanded action space for better control granularity
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(self.state_dim, self.action_dim).to(self.device)
        
        # Load the trained model with better error handling
        try:
            if os.path.exists(self.model_file):
                self.network.load_state_dict(torch.load(self.model_file, map_location=self.device))
                print(f"Successfully loaded model from {self.model_file}")
            else:
                print(f"Warning: Model file {self.model_file} not found. Using untrained model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model as fallback.")
        
        self.network.eval()  # Set to evaluation mode
        
        # Expanded action space for finer control
        self.forces = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]

    def select_action(self, state):
        '''
        Selects the best action according to the current policy.

        Args:
            state (numpy.ndarray): The current state [theta, theta_dot].

        Returns:
            int: Index of the best action.
        '''
        with torch.no_grad():  # Disable gradient computation
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            # Forward pass through the network
            probs, _ = self.network(state_tensor)
            
            # Select the action with highest probability
            action = torch.argmax(probs).item()
            
            return action

    def get_force(self, theta, theta_dot):
        '''
        Computes the control signal.

        Args:
            theta (float): Angle error signal.
            theta_dot (float): Angular velocity.

        Returns:
            float: Output signal (force).
        '''
        # Create state vector
        state = np.array([theta, theta_dot])
        
        # Select best action
        action_idx = self.select_action(state)
        
        # Get force from selected action
        force = self.forces[action_idx]
        
        # Apply small damping based on angular velocity for stability
        damping = -0.5 * theta_dot
        force += damping
        
        return force
        
# Instantiate A2C Controller
controller = A2CController()   

def controller_call(rad_big, theta_dot):
    '''
    Calls the A2C Controller to compute the control signal.

    Args:
        rad_big (float): Raw angle error signal.
        theta_dot (float): Angular velocity.

    Returns:
        float: Output signal (force).
    '''
    global controller

    # Normalize the angle (between -π and π)
    theta = (rad_big % (2*np.pi))
    if theta > np.pi:
        theta -= 2 * np.pi
        
    force = controller.get_force(theta, theta_dot)
    
    # Apply force limit for safety
    force = np.clip(force, -50.0, 50.0)
    
    return force