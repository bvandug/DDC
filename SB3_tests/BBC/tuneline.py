#!/usr/bin/env python3
import subprocess
import sys

def run_cmd(cmd):
    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

def run_tune(script, algo="dqn", trials=50, parallel=1, seed=42, device="cuda"):
    cmd = [
        sys.executable, script,
        "--algo", algo,
        "--n-trials", str(trials),
        "--n-parallel", str(parallel),
        "--seed", str(seed),
        "--device", device,
    ]
    run_cmd(cmd)

def run_train(script="jax_bbc_train.py", algo="dqn", timesteps=5_000_000,
              seed=42, n_envs=8, device="cuda", dqn_bins=41, noise="single"):
    cmd = [
        sys.executable, script,
        "--algo", algo,
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--n-envs", str(n_envs),
        "--device", device,
        "--dqn-bins", str(dqn_bins),
        "--noise", noise
    ]
    run_cmd(cmd)

if __name__ == "__main__":
    # Step 1: tune for 17 bins
    run_tune("tune_bbc_17.py", algo="dqn")

    # Step 2: tune for 41 bins
    run_tune("tune_bbc_41.py", algo="dqn")

    # Step 3: train with DQN (default 41 bins; change to 17 if you want)
    run_train("jax_bbc_train.py", algo="dqn", dqn_bins=41, noise="single")
