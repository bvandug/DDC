import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, TD3, PPO, SAC
import matlab.engine
import os

class ModelRunner:
    def __init__(self, model_path, model_name="PendCart", model_type="TD3", dt=0.01):
        """
        Initialize the model runner with a trained model and connect to Simulink.
        
        Args:
            model_path (str): Path to the saved .zip model file.
            model_name (str): The name of the Simulink model file (e.g., "PendCart").
            model_type (str): Type of model ("TD3", "A2C", "PPO", or "SAC").
            dt (float): The simulation time step.
        """
        self.model_name = model_name
        self.dt = dt
        self.max_force = 10.0
        self.current_time = 0.0

        # --- MATLAB Engine Setup ---
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()
        
        # Set up MATLAB path and load the Simulink model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.eng.addpath(script_dir, nargout=0)
        self.eng.load_system(self.model_name, nargout=0)
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
        print(f"Loaded Simulink model: {self.model_name}")
        
        # --- Model Loading ---
        print(f"Loading {model_type} model from {model_path}...")
        model_classes = {"TD3": TD3, "A2C": A2C, "PPO": PPO, "SAC": SAC}
        if model_type not in model_classes:
            raise ValueError(f"Model type must be one of {list(model_classes.keys())}")
        
        self.model = model_classes[model_type].load(model_path)
        
        # --- Visualization and Data ---
        print("Setting up visualization...")
        self.setup_visualization()
        self.reset_data_storage()
        
    def setup_visualization(self):
        """Set up the Matplotlib live plots."""
        plt.ion()
        self.fig, (self.ax_anim, self.ax_theta, self.ax_thetav, self.ax_u) = plt.subplots(4, 1, figsize=(6, 10))
        
        self.ax_anim.axis('off')
        self.pendulum_length = 1.0
        self.ax_anim.set_xlim(-self.pendulum_length, self.pendulum_length)
        self.ax_anim.set_ylim(-self.pendulum_length, self.pendulum_length)
        self.ax_anim.set_aspect('equal')
        
        self.line_rod, = self.ax_anim.plot([], [], lw=3)
        self.point_mass, = self.ax_anim.plot([], [], 'o', ms=8)
        self.line_theta = self.ax_theta.plot([], [], label="θ")[0]
        self.line_thetav = self.ax_thetav.plot([], [], label="θ̇")[0]
        self.line_u = self.ax_u.plot([], [], label="u")[0]
        
        for ax in (self.ax_theta, self.ax_thetav, self.ax_u):
            ax.set_xlabel("time (s)")
            ax.legend(loc="upper right")
            
    def reset_data_storage(self):
        """Reset all Python data storage lists."""
        self.times = []
        self.thetas = []
        self.thetavs = []
        self.actions = []

    def get_data(self):
        """Helper function to get data from the MATLAB workspace."""
        raw_ang = self.eng.eval("out.angle", nargout=1)
        raw_time = self.eng.eval("out.tout", nargout=1)
        
        # Handle single values
        if isinstance(raw_ang, float):
            angle_2d = [[raw_ang]]
        else:
            angle_2d = raw_ang

        if isinstance(raw_time, float):
            time_2d = [[raw_time]]
        else:
            time_2d = raw_time

        angle_lst = []
        for angle in angle_2d:
            angle_lst.append(angle[0])

        time_lst = []
        for t in time_2d:
            time_lst.append(t[0])
        
        return angle_lst, time_lst

    def reset_simulink(self, initial_angle):
        """Resets the Simulink model to a given initial state."""
        print(f"Resetting Simulink model to initial_angle: {initial_angle:.2f}")
        # Stop any previous simulation
        self.eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)
        self.current_time = 0.0
        
        # Set the initial angle in the model
        self.eng.set_param(f'{self.model_name}/Pendulum and Cart', 'init', str(initial_angle), nargout=0)
        
        # Set random noise seeds/power
        noise_seed = str(np.random.randint(1, 40000))
        noise_seed_v = str(np.random.randint(1, 40000))
        for blk, seed in [('Noise', noise_seed), ('Noise_v', noise_seed_v)]:
            self.eng.set_param(f'{self.model_name}/{blk}', 'seed', f'[{seed}]', nargout=0)
            self.eng.set_param(f'{self.model_name}/{blk}', 'Cov', '[0]', nargout=0)
        
        # Configure simulation to save its final state
        self.eng.set_param(self.model_name, 'FastRestart', 'off', 'LoadInitialState', 'off', nargout=0)
        
        # Run a tiny simulation to generate the initial state vector 'xFinal'
        self.eng.eval(
            f"out = sim('{self.model_name}', 'StopTime','1e-4', 'SaveFinalState','on', 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0
        )
        
        # Re-enable FastRestart for subsequent steps
        self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
        
        # Get the initial state from this short run
        angle_lst, time_lst = self.get_data()
        theta0 = angle_lst[-1]
        if len(angle_lst) >= 2:
            dt = time_lst[-1] - time_lst[-2]
            vel0 = (angle_lst[-1] - angle_lst[-2]) / (dt or self.dt)
        else:
            vel0 = 0.0
        
        return np.array([theta0, vel0], dtype=np.float32)

    def run_episode(self, initial_angle=None, max_time=5.0):
        """Run a single episode with the trained model on the Simulink simulation."""
        self.reset_data_storage()
        
        # Set a random initial angle if none is provided
        if initial_angle is None:
            initial_angle = np.random.uniform(-1, 1)
            while -0.05 < initial_angle < 0.05:
                initial_angle = np.random.uniform(-1, 1)
                
        # Reset the Simulink model and get the initial observation
        obs = self.reset_simulink(initial_angle)
        
        print("Starting episode...")
        while self.current_time < max_time:
            # Get action from the stable-baselines3 model
            action_raw, _ = self.model.predict(obs, deterministic=True)
            u = float(np.clip(action_raw[0], -self.max_force, self.max_force))
            
            # Apply the action to the Simulink model
            self.eng.set_param(f"{self.model_name}/Constant", 'Value', str(u), nargout=0)
            
            # Turn FastRestart OFF for this one sim
            self.eng.set_param(self.model_name, 'FastRestart', 'off', nargout=0)
            
            # Run the simulation for one time step
            stop_time = self.current_time + self.dt
            self.eng.eval(
                f"out = sim('{self.model_name}',"
                f" 'LoadInitialState','on',"
                f" 'InitialState','xFinal',"
                f" 'StopTime','{stop_time}',"
                f" 'SaveFinalState','on',"
                f" 'StateSaveName','xFinal');"
                "xFinal = out.xFinal;",
                nargout=0
            )
            
            # Re-enable FastRestart for speed
            self.eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)
            
            # Get the new state from the simulation
            angle_lst, time_lst = self.get_data()
            theta = angle_lst[-1]
            t = time_lst[-1]
            
            if len(angle_lst) >= 2:
                dt = t - time_lst[-2]
                theta_v = (theta - angle_lst[-2]) / (dt or self.dt)
            else:
                theta_v = 0.0
            
            # Update observation and store data
            obs = np.array([theta, theta_v], dtype=np.float32)
            self.current_time = t
            
            self.times.append(self.current_time)
            self.thetas.append(theta)
            self.thetavs.append(theta_v)
            self.actions.append(u)
            
            # Update visualization
            self.update_visualization()
            
            # End episode if pendulum has fallen
            if abs(theta) > np.pi / 2:
                print("Pendulum fell. Ending episode.")
                break
        
        return self.times, self.thetas, self.thetavs, self.actions
        
    def update_visualization(self):
        """Update the Matplotlib plots with the latest data."""
        if not self.times: return
            
        x = self.pendulum_length * np.sin(self.thetas[-1])
        y = -self.pendulum_length * np.cos(self.thetas[-1])
        
        self.line_theta.set_data(self.times, self.thetas)
        self.line_thetav.set_data(self.times, self.thetavs)
        self.line_u.set_data(self.times, self.actions)
        self.line_rod.set_data([0, x], [0, y])
        self.point_mass.set_data([x], [y])
        
        for ax in (self.ax_theta, self.ax_thetav, self.ax_u):
            ax.relim()
            ax.autoscale_view()
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        """Clean up resources by closing the plot and MATLAB engine."""
        print("Closing resources...")
        plt.close(self.fig)
        self.eng.quit()

if __name__ == "__main__":
    # --- Example Usage ---
    # Make sure 'td3_simulinker.zip' is in the same directory as this script
    model_file = "td3_simulinker500.zip"
    if not os.path.exists(model_file):
        print(f"Error: Model file not found at '{model_file}'")
        print("Please ensure the trained model is in the correct location.")
    else:
        runner = ModelRunner(model_file, model_type="TD3")
        try:
            # Run 3 episodes with different random starting angles
            for i in range(3):
                print(f"\n--- Running Episode {i+1} ---")
                runner.run_episode()
                print(f"Episode {i+1} completed.")
                plt.pause(1) # Pause for a moment before the next run
        finally:
            runner.close()