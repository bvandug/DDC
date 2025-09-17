import os
import sys
import subprocess
import shutil

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
IP_DIR = os.path.normpath(os.path.join(ROOT, "..", "Inverted_Pendulum"))
TRAIN_DIR = os.path.join(ROOT, "CP_JAX")

TRAIN_SCRIPT   = "cp_jax_train.py"
HP_SCRIPT      = "cp_jax_hp.py"
CP_EVAL_SCRIPT = os.path.join(ROOT, "cp_eval.py")
IP_EVAL_SCRIPT = os.path.join(IP_DIR, "ip_eval_1.py")

# Config
TIMESTEPS     = 100_000
ALGOS         = ["a2c","td3","sac","ddpg","ppo","dqn"]
NOISE_LEVELS  = [0, 0.001, 0.01, 0.1]
EVAL_EPISODES = 5

def run(cmd, cwd=None):
    """ Run a command synchronously, echoing it, and fail on non-zero exit.

        Parameters
        ----------
        cmd : Sequence[str | os.PathLike]
            Command and arguments, e.g. [sys.executable, "script.py", "--flag"].
        cwd : str | os.PathLike | None
            Working directory to execute the command in.

        Raises
        ------
        subprocess.CalledProcessError
            If the command exits with a non-zero status.
    """

    print("▶▶", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def _ps_dq(s: str) -> str:
    """ Return a PowerShell-safe double-quoted argument string.

        Escapes any embedded double quotes using PowerShell's escape character
        (backtick) and wraps the entire argument in double quotes. Useful when
        building a `-Command` string for PowerShell.

        Parameters
        ----------
        s : str
            Raw argument.

        Returns
        -------
        str
            Escaped and quoted argument suitable for PowerShell.
    """

    return '"' + str(s).replace('"', '`"') + '"'

def spawn_powershell_window(cmd_list, cwd: str):
    """ Launch a new PowerShell window and run a Python command in `cwd`.

        Builds a `-Command` that changes directory to `cwd` and executes the
        Python interpreter with the provided arguments. Attempts to find
        `powershell.exe` or `pwsh.exe`. The new console closes when the child
        command exits.

        Parameters
        ----------
        cmd_list : Sequence[str]
            Full command list; `cmd_list[0]` must be the Python executable
            (e.g., `sys.executable`), followed by script and args.
        cwd : str
            Working directory for the command.

        Returns
        -------
        subprocess.Popen
            Handle to the spawned PowerShell process.

        Notes
        -----
        - Uses `CREATE_NEW_CONSOLE` to open a separate terminal window.
        - Primarily intended for Windows; behavior on other OSes may differ.
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



    # run([sys.executable, HP_SCRIPT],cwd=TRAIN_DIR)
                
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
