import numpy as np
from stable_baselines3 import PPO, SAC, TD3

# Don’t load on import
_model = None
MODEL_NAME = "td3_simulinker"  # base name of your saved model

def _get_model():
    global _model
    if _model is None:
        _model = TD3.load(MODEL_NAME)
    return _model

def controller_call(theta, theta_v, t):
    model = _get_model()
    obs = np.array([theta, theta_v], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return float(action[0])

print(controller_call(0.1, 0.0, 0))
print(controller_call(0.5, 0.0, 0))
print(controller_call(-0.5, 0.0, 0))
