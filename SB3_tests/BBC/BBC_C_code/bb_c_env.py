# buckboost_c_env.py
import ctypes, os, sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

def _load_lib():
    here = Path(__file__).resolve().parent
    names = ["bb_core.dll", "libbb_core.so", "libbb_core.dylib"]
    for n in names:
        p = here / n
        if p.exists():
            return ctypes.CDLL(str(p))
    # If not found, try system paths
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            pass
    raise OSError("Could not find bb_core shared library")

_lib = _load_lib()

# C signatures
_lib.bb_create.restype = ctypes.c_void_p
_lib.bb_create.argtypes = [
    ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
]
_lib.bb_destroy.restype = None
_lib.bb_destroy.argtypes = [ctypes.c_void_p]

_lib.bb_reset.restype = None
_lib.bb_reset.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]

_lib.bb_step.restype = None
_lib.bb_step.argtypes = [
    ctypes.c_void_p, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
]

class BuckBoostCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
        dt=5e-6, frame_skip=26, max_episode_steps=4000, grace_period_steps=100,
        target_voltage=-30.0,
        Vin=48.0, L=470e-6, C=220e-6, R_load=20.0,
        Ron_sw=0.1, R_L=0.0, Vf=0.7, Ron_d=0.001, R_C=0.0,
        G_off=0.0, enable_rr=False, Qrr=0.0, trr=0.0,
        duty_min=0.1, duty_max=0.9,
    ):
        super().__init__()
        self.ptr = ctypes.c_void_p(_lib.bb_create(
            ctypes.c_double(dt), ctypes.c_int(frame_skip),
            ctypes.c_int(max_episode_steps), ctypes.c_int(grace_period_steps),
            ctypes.c_double(target_voltage),
            ctypes.c_double(Vin), ctypes.c_double(L), ctypes.c_double(C), ctypes.c_double(R_load),
            ctypes.c_double(Ron_sw), ctypes.c_double(R_L),
            ctypes.c_double(Vf), ctypes.c_double(Ron_d), ctypes.c_double(R_C),
            ctypes.c_double(G_off), ctypes.c_int(1 if enable_rr else 0),
            ctypes.c_double(Qrr), ctypes.c_double(trr),
            ctypes.c_double(duty_min), ctypes.c_double(duty_max),
        ))
        if not self.ptr:
            raise RuntimeError("bb_create failed")

        self.action_space = spaces.Box(
            low=np.array([duty_min], np.float32),
            high=np.array([duty_max], np.float32),
            dtype=np.float32,
        )
        high = np.array([np.finfo(np.float32).max]*4, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._obs_buf = (ctypes.c_double * 4)()
        self._rew = ctypes.c_double(0.0)
        self._term = ctypes.c_int(0)
        self._trunc = ctypes.c_int(0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        _lib.bb_reset(self.ptr, self._obs_buf)
        obs = np.frombuffer(self._obs_buf, dtype=np.float64, count=4).astype(np.float32)
        return obs, {}

    def step(self, action):
        a = float(action[0]) if isinstance(action, (np.ndarray, list, tuple)) else float(action)
        _lib.bb_step(self.ptr, ctypes.c_double(a), self._obs_buf, ctypes.byref(self._rew),
                     ctypes.byref(self._term), ctypes.byref(self._trunc))
        obs = np.frombuffer(self._obs_buf, dtype=np.float64, count=4).astype(np.float32)
        return obs, float(self._rew.value), bool(self._term.value), bool(self._trunc.value), {}

    def close(self):
        if getattr(self, "ptr", None):
            _lib.bb_destroy(self.ptr)
            self.ptr = None
        super().close()


# if __name__ == "__main__":
#     import numpy as np
#     env = BuckBoostCEnv()
#     obs, _ = env.reset()
#     total = 0.0
#     for t in range(100000):
#         action = np.array([0.35], dtype=np.float32)
#         obs, r, term, trunc, _ = env.step(action)
#         total += r
#         if term or trunc:
#             break
#     print("Steps:", t+1, "Return:", total)
#     env.close()
