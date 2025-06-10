# simulink_env.py

import gym
from gym import spaces
import numpy as np
import matlab.engine

# ────────────────────────────────────────────────────────────────────────────────
# Start MATLAB engine exactly once, at import time, and bind it to the module
# name `eng`.  Now every function in this file can refer to that same engine.
# ────────────────────────────────────────────────────────────────────────────────

class SimulinkEnv(gym.Env):
    """
    A Gym wrapper around your Pendulum-on-Cart Simulink model,
    now using the global `eng` and the unmodified get_data().
    """
    metadata = {'render.modes': []}

    def __init__(self,
             model_name: str = "PendCart",
             agent_block: str = "PendCart/DRL",
             dt: float = 0.02,
             max_episode_time: float = 5,
             angle_threshold: float = np.pi/2):
        super().__init__()
        print("\nInitializing Simulink Environment...")
        global eng
        print("Starting MATLAB engine...")
        eng = matlab.engine.start_matlab()
        print("Loading Simulink model...")
        eng.load_system(model_name, nargout=0)
        print("Enabling FastRestart for better performance...")
        eng.set_param('PendCart', 'FastRestart', 'on', nargout=0)
        
        # Disable warnings about parameter changes
        eng.eval("warning('off', 'Simulink:Engine:ModelOperatingPointChanged');", nargout=0)

        self.model_name       = model_name
        self.agent_block      = agent_block
        self.dt               = dt
        self.current_time     = 0.0
        self.max_episode_time = max_episode_time
        self.angle_threshold  = angle_threshold
        self.pendulum_length  = 1.0  # Length of the pendulum for visualization

        print("Setting up action and observation spaces...")
        max_force = 10.0
        self.action_space = spaces.Box(
            low=-max_force, high=+max_force,
            shape=(1,), dtype=np.float32
        )
        high = np.array([np.pi, np.finfo(np.float32).max], np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32
        )

        print("Setting initial conditions...")
        # Set random initial angle
        initial_angle = np.random.uniform(-1, 1)
        while -0.05 <= initial_angle <= 0.05:
            initial_angle = np.random.uniform(-1, 1)
        eng.set_param(f'{model_name}/Pendulum and Cart', 'init', str(initial_angle), nargout=0)

        # Set random noise
        print("Configuring noise parameters...")
        noise_seed   = str(np.random.randint(1, high=40000))
        noise_power  = 0
        eng.set_param(f'{model_name}/Noise',   'seed', f'[{noise_seed}]', nargout=0)
        eng.set_param(f'{model_name}/Noise',   'Cov',  f'[{noise_power}]',  nargout=0)
        noise_seed_v = str(np.random.randint(1, high=40000))
        noise_power_v = 0
        eng.set_param(f'{model_name}/Noise_v', 'seed', f'[{noise_seed_v}]', nargout=0)
        eng.set_param(f'{model_name}/Noise_v', 'Cov',  f'[{noise_power_v}]', nargout=0)
        print("Environment initialization complete!\n")

    def get_data(self):
        """
        Collects and returns the workspace data, handling both
        single-value and multi-value returns from MATLAB.
        Returns:
            angle_lst (List[float]) : Observed angles.
            time_lst  (List[float]) : Simulation times.
        """
        # pull back whatever MATLAB has in out.angle / out.tout
        raw_ang  = eng.eval("out.angle", nargout=1)
        raw_time = eng.eval("out.tout",  nargout=1)

        # if it's a bare float, wrap it so our loop works
        if isinstance(raw_ang, float):
            angle_2d = [[raw_ang]]
        else:
            angle_2d = raw_ang

        if isinstance(raw_time, float):
            time_2d = [[raw_time]]
        else:
            time_2d = raw_time

        # now your original two-loops will never see a float
        angle_lst = []
        for angle in angle_2d:
            # angle might be [val], so angle[0] is safe
            angle_lst.append(angle[0])

        time_lst = []
        for t in time_2d:
            time_lst.append(t[0])

        return angle_lst, time_lst

    def reset(self):
        print("\nResetting environment...")
        # stop any sim and zero the clock
        self.current_time = 0.0
        eng.set_param(self.model_name, 'SimulationCommand', 'stop', nargout=0)

        # Set random initial angle
        print("Setting new initial angle...")
        initial_angle = np.random.uniform(-1, 1)
        while -0.05 < initial_angle < 0.05:
            initial_angle = np.random.uniform(-1, 1)
        eng.set_param(f'{self.model_name}/Pendulum and Cart', 'init', str(initial_angle), nargout=0)

        # Set random noise seeds/power
        print("Updating noise parameters...")
        noise_seed    = str(np.random.randint(1, 40000))
        noise_seed_v  = str(np.random.randint(1, 40000))
        for blk, seed in [( 'Noise', noise_seed ), ( 'Noise_v', noise_seed_v )]:
            eng.set_param(f'{self.model_name}/{blk}', 'seed', f'[{seed}]', nargout=0)
            eng.set_param(f'{self.model_name}/{blk}', 'Cov',  '[0]',          nargout=0)

        # temporarily disable FastRestart and any LoadInitialState
        print("Preparing simulation state...")
        eng.set_param(self.model_name,
                      'FastRestart',       'off',
                      'LoadInitialState',  'off',
                      nargout=0)
        
        # run zero-length sim to seed the state and create 'xFinal'
        print("Initializing simulation...")
        eng.eval(
        f"out = sim('{self.model_name}',"
        " 'StopTime','1e-4',"
        " 'SaveFinalState','on',"
        " 'StateSaveName','xFinal');"
        "xFinal = out.xFinal;",
        nargout=0
        )

        # Re-enable FastRestart for speed on subsequent steps
        eng.set_param(self.model_name, 'FastRestart', 'on', nargout=0)

        # now pull initial angle & velocity
        print("Getting initial state...")
        angle_lst, time_lst = self.get_data()
        theta0 = angle_lst[-1]
        if len(angle_lst) >= 2:
            dt   = time_lst[-1] - time_lst[-2]
            vel0 = (angle_lst[-1] - angle_lst[-2]) / (dt or self.dt)
        else:
            vel0 = 0.0

        print(f"Reset complete. Initial angle: {theta0:.3f}, Initial velocity: {vel0:.3f}\n")
        return np.array([theta0, vel0], dtype=np.float32)

    def step(self, action):
        # 1. clip & apply action
        u = float(np.clip(action,
                            self.action_space.low,
                            self.action_space.high))
        eng.set_param(f"{self.model_name}/Constant",
                        'Value', str(u),
                        nargout=0)

        # 2. decide next stop time
        start = self.current_time
        stop  = start + self.dt

        # 3. turn FastRestart OFF for this one sim  
        eng.set_param(self.model_name,
                        'FastRestart', 'off',
                        nargout=0)

        # 4. run exactly one step
        eng.eval(
        f"out = sim('{self.model_name}',"
        f" 'LoadInitialState','on',"
        f" 'InitialState','xFinal',"
        f" 'StopTime','{stop}',"
        f" 'SaveFinalState','on',"
        f" 'StateSaveName','xFinal');"
        "xFinal = out.xFinal;",
        nargout=0
        )

        # 5. immediately re-enable FastRestart for speed
        eng.set_param(self.model_name,
                        'FastRestart', 'on',
                        nargout=0)

        # 6. grab the latest angle & time
        angle_lst, time_lst = self.get_data()
        theta = angle_lst[-1]
        t     = time_lst[-1]

        # 7. compute velocity
        if len(angle_lst) >= 2:
            dt   = time_lst[-1] - time_lst[-2]
            vel  = (angle_lst[-1] - angle_lst[-2]) / (dt or self.dt)
        else:
            vel  = 0.0

        # 8. update clock
        self.current_time = t

        # 9. compute reward
        reward = -abs(theta) - 0.1*abs(vel) - 0.001*abs(u)

        # 10. check if episode is done
        done = (t >= self.max_episode_time)

        return np.array([theta, vel], dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        print("\nClosing environment and MATLAB engine...")
        # Re-enable warnings before closing
        eng.eval("warning('on', 'Simulink:Engine:ModelOperatingPointChanged');", nargout=0)
        eng.quit()
        print("Environment closed successfully.\n")
