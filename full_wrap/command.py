#!/usr/bin/env python3
import os
import subprocess
import sys

# 1) Where this driver lives (parent of ip_jax/)
ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "ip_jax")
TRAIN_SCRIPT = "ip_jax_train.py"
EVAL_SCRIPT  = os.path.join(ROOT, "combinedeval.py")

TIMESTEPS    = 100_000
ALGOS        = ["td3", "a2c", "sac", "ddpg", "ppo", "dqn"]
NOISE_LEVELS = [0.01, 0.1]

def run(cmd, cwd=None):
    print("▶▶", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

if __name__ == "__main__":
    # # 2) Run trainings inside ip_jax/
    # for algo in ALGOS:
    #     run([sys.executable, TRAIN_SCRIPT,
    #          "--algo", algo, "--timesteps", str(TIMESTEPS)],
    #         cwd=TRAIN_DIR)

    for nl in NOISE_LEVELS:
        for algo in ALGOS:
            run([sys.executable, TRAIN_SCRIPT,
                 "--algo", algo,
                 "--timesteps", str(TIMESTEPS),
                 "--noise",
                 "--noise-level", str(nl)],
                cwd=TRAIN_DIR)

    # 3) Back to root, run combined evaluation
    run([sys.executable, EVAL_SCRIPT], cwd=ROOT)
