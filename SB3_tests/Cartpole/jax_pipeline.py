#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil

# === Paths (script lives in SB3_tests/Cartpole) ===
ROOT = os.path.dirname(os.path.abspath(__file__))              # .../SB3_tests/Cartpole
IP_DIR = os.path.normpath(os.path.join(ROOT, "..", "Inverted_Pendulum"))
TRAIN_DIR = os.path.join(ROOT, "CP_JAX")

TRAIN_SCRIPT   = "cp_jax_train.py"
CP_EVAL_SCRIPT = os.path.join(ROOT, "cp_eval.py")
IP_EVAL_SCRIPT = os.path.join(IP_DIR, "ip_eval_1.py")

# === Config ===
TIMESTEPS     = 100_000
ALGOS         = ["a2c","td3","sac","ddpg","ppo","dqn"]
NOISE_LEVELS  = [0, 0.001, 0.01, 0.1]
EVAL_EPISODES = 5

def run(cmd, cwd=None):
    print("▶▶", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def _ps_dq(s: str) -> str:
    """PowerShell-safe double-quote for an argument."""
    return '"' + str(s).replace('"', '`"') + '"'

def spawn_powershell_window(cmd_list, cwd: str):
    """
    Open a new PowerShell window and run the given command in that directory.
    Returns the Popen handle so we can wait() for it to finish.
    The window will close automatically when the command completes.
    """
    ps = shutil.which("powershell.exe") or shutil.which("pwsh.exe") or "powershell.exe"
    python = cmd_list[0]
    args   = " ".join(_ps_dq(a) for a in cmd_list[1:])
    ps_command = f'cd {_ps_dq(cwd)}; & {_ps_dq(python)} {args}'
    return subprocess.Popen([ps, "-Command", ps_command],
                            creationflags=subprocess.CREATE_NEW_CONSOLE)

if __name__ == "__main__":
    # === 1) Inverted Pendulum evals first (4 noises) ===
    ip_procs = []
    for nl in NOISE_LEVELS:
        eval_cmd = [sys.executable, IP_EVAL_SCRIPT, "--algo", "all", "--episodes", str(EVAL_EPISODES)]
        if nl != 0:
            eval_cmd += ["--env-noise", f"{nl:.3f}"]
        print(f"Launching IP eval PowerShell for env-noise={nl:.3f} …")
        ip_procs.append(spawn_powershell_window(eval_cmd, cwd=IP_DIR))

    print("Waiting for all IP eval windows to finish…")
    for p in ip_procs:
        p.wait()
    print("All IP evals completed.\n")

    # === 2) Cartpole trainings (algo × noise) ===
    for nl in NOISE_LEVELS:
        for algo in ALGOS:
            run([sys.executable, TRAIN_SCRIPT,
                 "--algo", algo,
                 "--timesteps", str(TIMESTEPS),
                 "--noise",
                 "--noise-level", str(nl)],
                cwd=TRAIN_DIR)

    # === 3) Cartpole evals (4 noises) ===
    cp_procs = []
    for nl in NOISE_LEVELS:
        eval_cmd = [sys.executable, CP_EVAL_SCRIPT, "--algo", "all", "--episodes", str(EVAL_EPISODES)]
        if nl != 0:
            eval_cmd += ["--env-noise", f"{nl:.3f}"]  # omit for 0 (default)
        print(f"Launching CP eval PowerShell for env-noise={nl:.3f} …")
        cp_procs.append(spawn_powershell_window(eval_cmd, cwd=ROOT))

    print("Waiting for all CP eval windows to finish…")
    for p in cp_procs:
        p.wait()
    print("All CP evals completed.")
