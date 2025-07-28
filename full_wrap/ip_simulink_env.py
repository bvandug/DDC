import gym
from gym import spaces
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt


# >>> ADDED FOR DQN --------------------------------------------------------------
class DiscretizedActionWrapper(gym.ActionWrapper):
    """
    Wraps a continuous-action env so a discrete index is mapped to a pre-defined
    force value.  Suitable for running SB3-DQN on SimulinkEnv.
    """

    def __init__(self, env, force_values):
        super().__init__(env)
        self.force_values = np.asarray(force_values, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.force_values))

    def action(self, act_idx):
        """Convert the integer chosen by DQN into the continuous force."""
        return np.array([self.force_values[int(act_idx)]], dtype=np.float32)


# ------------------------------------------------------------------------------


class SimulinkEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        model_name: str = "pendulum",
        # agent_block: str = "PendCart/DRL",
        dt: float = 0.01,
        max_episode_time: float = 5,
        angle_threshold: float = np.pi / 2,
    ):
        super().__init__()

        import shutil
        import tempfile
        import uuid
        import os

        # 🔄 Instance-specific MATLAB engine
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab("-nodesktop -licmode onlinelicensing")

        # 🆕 Create a unique copy of the model file
        unique_id = uuid.uuid4().hex[:8]
        self.model_name = f"{model_name}_{unique_id}"
        self.model_path = os.path.join(tempfile.gettempdir(), f"{self.model_name}.slx")
        shutil.copy(f"{model_name}.slx", self.model_path)

        # Load the unique model copy
        self.eng.load_system(self.model_path, nargout=0)
        self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

        # self.agent_block = agent_block
        self.dt = dt
        self.current_time = 0.0
        self.max_episode_time = max_episode_time
        self.angle_threshold = angle_threshold
        self.pendulum_length = 0.15

        max_torque = 2.0  # same as ip_jax.PendulumConfig.max_torque
        self.action_space = spaces.Box(
            low=-max_torque, high=+max_torque, shape=(1,), dtype=np.float32
        )

        high = np.array([np.pi, np.finfo(np.float32).max], np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Random init angle
        initial_angle = np.random.uniform(-1, 1)
        while -0.05 <= initial_angle <= 0.05:
            initial_angle = np.random.uniform(-1, 1)
        self.eng.set_param(
            f"{self.model_name}/Pendulum and Cart",
            "init",
            str(initial_angle),
            nargout=0,
        )

        # # Random noise
        # noise_seed = str(np.random.randint(1, 40000))
        # noise_power = 0
        # self.eng.set_param(f"{self.model_name}/Noise", "seed", f"[{noise_seed}]", nargout=0)
        # self.eng.set_param(f"{self.model_name}/Noise", "Cov", f"[{noise_power}]", nargout=0)
        # noise_seed_v = str(np.random.randint(1, 40000))
        # noise_power_v = 0
        # self.eng.set_param(
        #     f"{self.model_name}/Noise_v", "seed", f"[{noise_seed_v}]", nargout=0
        # )
        # self.eng.set_param(
        #     f"{self.model_name}/Noise_v", "Cov", f"[{noise_power_v}]", nargout=0
        # )

    def get_data(self):
        # pull _both_ angle and true angular velocity out of Simulink
        raw_ang = self.eng.eval("out.angle", nargout=1)
        raw_vel = self.eng.eval("out.angle_v", nargout=1)
        raw_time = self.eng.eval("out.tout", nargout=1)

        # flatten
        ang2d = [[raw_ang]] if isinstance(raw_ang, float) else raw_ang
        vel2d = [[raw_vel]] if isinstance(raw_vel, float) else raw_vel
        t2d = [[raw_time]] if isinstance(raw_time, float) else raw_time

        angle_lst = [a[0] for a in ang2d]
        vel_lst = [v[0] for v in vel2d]
        time_lst = [t[0] for t in t2d]
        return angle_lst, vel_lst, time_lst

    def reset(self):

        self.current_time = 0.0
        self.eng.set_param(self.model_name, "SimulationCommand", "stop", nargout=0)

        initial_angle = np.random.uniform(-1, 1)
        while -0.05 < initial_angle < 0.05:
            initial_angle = np.random.uniform(-1, 1)
        # self.eng.set_param(
        #     f"{self.model_name}/Pendulum and Cart",
        #     "init",
        #     str(initial_angle),
        #     nargout=0,
        # )

        # noise_seed = str(np.random.randint(1, 40000))
        # noise_seed_v = str(np.random.randint(1, 40000))
        # for blk, seed in [("Noise", noise_seed), ("Noise_v", noise_seed_v)]:
        #     self.eng.set_param(
        #         f"{self.model_name}/{blk}", "seed", f"[{seed}]", nargout=0
        #     )
        #     self.eng.set_param(f"{self.model_name}/{blk}", "Cov", "[0]", nargout=0)

        self.eng.set_param(
            self.model_name, "FastRestart", "off", "LoadInitialState", "off", nargout=0
        )
        self.eng.eval(
            f"out = sim('{self.model_name}', 'StopTime','1e-4', 'SaveFinalState','on', 'StateSaveName','xFinal'); xFinal = out.xFinal;",
            nargout=0,
        )
        self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

        # angle_lst, vel_lst, time_lst = self.get_data()
        # theta = angle_lst[-1]
        # t = time_lst[-1]
        # vel = vel_lst[-1]

        return np.array([intial_angle, 0.0], dtype=np.float32)

    def step(self, action):
        torque = float(np.clip(action, self.action_space.low, self.action_space.high))
        # and then send `torque` to the Constant block:
        self.eng.set_param(
            f"{self.model_name}/Constant", "Value", str(torque), nargout=0
        )

        start, stop = self.current_time, self.current_time + self.dt
        self.eng.set_param(self.model_name, "FastRestart", "off", nargout=0)
        self.eng.eval(
            f"out = sim('{self.model_name}',"
            f" 'LoadInitialState','on',"
            f" 'InitialState','xFinal',"
            f" 'StopTime','{stop}',"
            f" 'SaveFinalState','on',"
            f" 'StateSaveName','xFinal');"
            "xFinal = out.xFinal;",
            nargout=0,
        )
        self.eng.set_param(self.model_name, "FastRestart", "on", nargout=0)

        angle_lst, vel_lst, time_lst = self.get_data()
        theta = angle_lst[-1]
        t = time_lst[-1]
        vel = vel_lst[-1]
        obs = np.array([theta, vel], dtype=np.float32)
        # print(obs)

        # MODIFICATION: Update penalties to match the new training reward
        # position_reward = np.cos(theta)
        # velocity_penalty = 0.05 * vel**2
        # effort_penalty = 0.005 * (action / self.action_space.high[0])**2
        reward = np.cos(theta)

        # print(f"Reward: {reward}")

        done = abs(theta) > self.angle_threshold or t >= self.max_episode_time
        self.current_time = t

        return obs, reward, done, {"time": t}

    def render(self, mode="human"):
        pass

    def close(self):
        import os
        import shutil
        import glob

        self.eng.quit()

        # Clean up temporary model file
        if hasattr(self, "model_path") and os.path.exists(self.model_path):
            try:
                os.remove(self.model_path)
                print(f"Deleted temporary model file: {self.model_path}")
            except Exception as e:
                print(f"Warning: could not delete model file: {e}")

        # Clean slprj/<model_name> folder (worker-specific)
        slprj_model_dir = os.path.join(os.getcwd(), "slprj", self.model_name)
        if os.path.exists(slprj_model_dir):
            try:
                shutil.rmtree(slprj_model_dir)
                print(f"Deleted slprj cache for model: {self.model_name}")
            except Exception as e:
                print(f"Warning: could not delete slprj folder: {e}")

        # Remove any autosave or .slxc file linked to this model
        base_name = os.path.splitext(self.model_name)[0]
        autosave_file = f"{base_name}.slx.autosave"
        slxc_file = f"{base_name}.slxc"

        for filename in [autosave_file, slxc_file]:
            full_path = os.path.join(os.getcwd(), filename)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    print(f"Deleted: {full_path}")
                except Exception as e:
                    print(f"Warning: could not delete file: {full_path}: {e}")
