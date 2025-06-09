import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, TD3, PPO, SAC
import matlab.engine
import os

class ModelRunner:
    def __init__(self, model_path, model_type="TD3"):
        """
        Initialize the model runner with a trained model.
        
        Args:
            model_path (str): Path to the saved model
            model_type (str): Type of model ("TD3", "A2C", "PPO", or "SAC")
        """
        # Start MATLAB engine
        print("Starting MATLAB engine")
        self.eng = matlab.engine.start_matlab()
        
        # Set up MATLAB path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.eng.cd(script_dir, nargout=0)
        
        # Load the model
        print("Loading model")
        model_classes = {
            "TD3": TD3,
            "A2C": A2C,
            "PPO": PPO,
            "SAC": SAC
        }
        if model_type not in model_classes:
            raise ValueError(f"Model type must be one of {list(model_classes.keys())}")
        
        self.model = model_classes[model_type].load(model_path)
        
        # Initialize visualization
        print("Setting up visualization")
        self.setup_visualization()
        
        # Initialize data storage
        self.reset_data()
        
    def setup_visualization(self):
        """Set up the visualization plots"""
        plt.ion()
        self.fig, (self.ax_anim, self.ax_theta, self.ax_thetav, self.ax_u) = plt.subplots(4, 1, figsize=(6, 10))
        
        # Set up animation axes
        self.ax_anim.axis('off')
        self.pendulum_length = 1.0
        self.ax_anim.set_xlim(-self.pendulum_length, self.pendulum_length)
        self.ax_anim.set_ylim(-self.pendulum_length, self.pendulum_length)
        self.ax_anim.set_aspect('equal')
        
        # Create empty Line2D objects
        self.line_rod, = self.ax_anim.plot([], [], lw=3)
        self.point_mass, = self.ax_anim.plot([], [], 'o', ms=8)
        self.line_theta = self.ax_theta.plot([], [], label="θ")[0]
        self.line_thetav = self.ax_thetav.plot([], [], label="θ̇")[0]
        self.line_u = self.ax_u.plot([], [], label="u")[0]
        
        for ax in (self.ax_theta, self.ax_thetav, self.ax_u):
            ax.set_xlabel("time (s)")
            ax.legend(loc="upper right")
            
    def reset_data(self):
        """Reset all data storage"""
        self.times = []
        self.thetas = []
        self.thetavs = []
        self.actions = []
        self.current_time = 0.0
        
    def run_episode(self, initial_angle=None, max_steps=500, dt=0.01):
        """
        Run a single episode with the trained model.
        
        Args:
            initial_angle (float, optional): Initial angle of the pendulum. If None, random.
            max_steps (int): Maximum number of steps to run
            dt (float): Time step size
        """
        self.reset_data()
        
        # Set initial angle
        if initial_angle is None:
            initial_angle = np.random.uniform(-1, 1)
            while -0.05 < initial_angle < 0.05:
                initial_angle = np.random.uniform(-1, 1)
                
        # Initialize state
        theta = initial_angle
        theta_v = 0.0
        obs = np.array([theta, theta_v], dtype=np.float32)
        
        # Run episode
        for step in range(max_steps):
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Store data
            self.times.append(self.current_time)
            self.thetas.append(theta)
            self.thetavs.append(theta_v)
            self.actions.append(action[0])
            
            # Update visualization
            self.update_visualization()
            
            # Simulate one step (you would need to implement this based on your system)
            # For now, we'll use a simple pendulum simulation
            theta_v = theta_v + dt * (np.sin(theta) + action[0])
            theta = theta + dt * theta_v
            
            # Update observation
            obs = np.array([theta, theta_v], dtype=np.float32)
            self.current_time += dt
            
            # Check if episode should end
            if abs(theta) > np.pi/2:
                break
                
        return self.times, self.thetas, self.thetavs, self.actions
        
    def update_visualization(self):
        """Update the visualization with current data"""
        if not self.times:
            return
            
        # Update pendulum position
        x = self.pendulum_length * np.sin(self.thetas[-1])
        y = -self.pendulum_length * np.cos(self.thetas[-1])
        
        # Update lines
        self.line_theta.set_data(self.times, self.thetas)
        self.line_thetav.set_data(self.times, self.thetavs)
        self.line_u.set_data(self.times, self.actions)
        self.line_rod.set_data([0, x], [0, y])
        self.point_mass.set_data([x], [y])
        
        # Rescale axes
        for ax in (self.ax_theta, self.ax_thetav, self.ax_u):
            ax.relim()
            ax.autoscale_view()
            
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        """Clean up resources"""
        plt.close(self.fig)
        self.eng.quit()

if __name__ == "__main__":
    # Example usage
    runner = ModelRunner("td3_simulinker.zip", model_type="TD3")
    try:
        # Run multiple episodes
        for i in range(3):
            print(f"\nRunning episode {i+1}")
            times, thetas, thetavs, actions = runner.run_episode()
            print(f"Episode {i+1} completed in {len(times)} steps")
    finally:
        runner.close() 