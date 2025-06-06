import numpy as np
from stable_baselines3 import PPO, SAC, TD3
import random

# Don’t load on import
_model = None
MODEL_NAME = "td3_simulink"  # base name of your saved model

def _get_model():
    global _model
    if _model is None:
        # this is where the zip must already exist
        _model = TD3.load(MODEL_NAME)
    return _model

def controller_call(theta, theta_v, t):
    model = _get_model()
    obs = np.array([theta, theta_v], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return float(action[0])

