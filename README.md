# Evaluating DRL for control systems
The project focuses on evaluating:
- The Stablebaselines3 DRL algorithms (PPO, A2C, DQN, DDPG, TD3, SAC) on various control systems against a PID baseline.
- Creating simulation environments that will train SB3 models faster but with similar fidelity.

## Authors
**Benjamin Ruijsch van Dugteren**<br>
*University of Cape Town* <br>
*RJSBEN001@myuct.ac.za*
<br>

**Nicholas Cristaudo**<br>
*University of Cape Town* <br>
*CRSNIC014@myuct.ac.za*

**Nathan Wells**<br>
*University of Cape Town* <br>
*WLLNAT033@myuct.ac.za*
___

# Project Allocation:
- **Nathan Wells**:
- **Ben Ruijsch Van Dugteren**:
- **Nicholas Cristaudo**:


# Full code base:
- The full code base is available at: https://github.com/bvandug/DDC
- This includes tuned models, evaluations, databases and tuning and hyperparameter logs.

# Structure for DRL Methods Explained

To explore the current DRL work, browse the `SB3_tests` folder in this repository. 
The PID benchmark work is done in `PID`.

## Table of Contents
- [Systems Included](#systems-included)
- [Repository Layout](#repository-layout)
- [MATLAB Dependency Setup](#matlab-dependency-setup)
- [Python Simulation Environments](#python-simulation-environments)
- [Running the Inverted Pendulum](#running-the-inverted-pendulum)
- [Running Cart-Pole](#running-cart-pole)
- [Running the Buck Converter](#running-the-buck-converter)
- [Running the Buck-boost Converter](#running-the-buck-boost-converter)
- [Acknowledgements](#acknowledgements)

## Systems Included

- **BC**: Buck Converter  
- **BBC**: Buck-Boost Converter  
- **Inverted Pendulum**
- **Cartpole**

## Repository Layout
- `README.md`: Project overview, usage instructions, and workflow summaries.
- `requirements.txt`: Python dependencies for the DRL experiments.
- `SB3_tests/`: Primary codebase for reinforcement-learning experiments.
  - `Inverted_Pendulum/`: JAX, NumPy, and MATLAB pipelines (`ip_jax/`, `IP_NUMPY/`, `IP_MATLAB/`) plus evaluation scripts.
  - `Cartpole/`: Matching JAX/NumPy/MATLAB stacks for the cart-pole system (`CP_JAX/`, `CP_NUMPY/`, `CP_MATLAB/`).
  - `BC/` & `BBC/`: Buck and buck-boost converter experiments, including Python envs, Simulink bridges, tuning utilities, and evaluation scripts.
  - `cp_verify_envs/`, `ip_verify_envs/`: Sanity-check environments and utilities used during development.
- `PID/`: Classical controller scripts and Simulink models for each plant.
- `venv/`: Optional local virtual environment (not required if you manage Python packages externally).

## MATLAB Dependency Setup
**This project is done with Matlab 2024b** -before setting up make sure you install MATLAB with the following packages:
- MATLAB Coder
- MATLAB Compiler
- MATLAB Compiler SDK
- Simulink
- Simscape
- Simscape Electrical
- Simulink Coder
- Simulink Compiler
Installation can be done from the official Mathworks website: https://ww2.mathworks.cn/help/install/ug/install-products-with-internet-connection.html
  
To get MATLAB dependency installed:
- Get the path for the folder you are coding/cloning to on your computer
- Run PowerShell in admin mode
- Navigate to where the main folder is
- Activate virtual environment
- Enter the following commands:
   - PS> cd "C:\Program Files\MATLAB\R2024b\extern\engines\python"- where your MATLAB is installed
   - PS> & "C:\... \MATLAB\venv\Scripts\python.exe" setup.py install - in your venv


## Python Simulation Environments

To run the Python simulation environments:

1. **Clone or pull** this repo so that you have the latest `SB3_tests` folder.  
2. **Create and activate** a virtual environment:

   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate
   ```

PID controllers are optional; you can skip the `PID/` workflows when you only need the DRL agents.
## Running the code

### Running the Inverted Pendulum

**Available Environments**
- `ip_jax/`: JAX environment used for the Optuna sweeps and the training of the models used in the project.
- `IP_NUMPY/`: Lightweight NumPy/Gym wrapper mirroring the JAX dynamics for quick local validation.
- `IP_MATLAB/` + `pendulum.slx`: MATLAB/Simulink plant for high-fidelity tests and PID benchmarking.
- `PID/`: Classical controller baselines that can be compared against the DRL agents.

You can train and evaluate agents in any of these environments; the results of the DRL models were trained in JAX/NumPy and evaluated in MATLAB.

**Workflow: JAX (reported results)**
1. (Optional) `python ip_jax_hp.py` reruns the Optuna sweeps and refreshes `ip_jax/jax_hp_results/`.
2. `python ip_jax_train.py --algos all --timesteps 100000` trains every configured agent on the fast JAX environment, saving to `ip_jax/jax_models/<algo>_noise_XXX/`. Use `--algos ppo sac` to limit the set or append `--noise --noise-level 0.01` to inject observation noise.
3. Evaluate the JAX checkpoints with `python ip_eval.py --algo all --root ip_jax/jax_models --episodes 5` and optional filters (`--env-noise 0.01`, `--name-contains noise_0.010`).

**Workflow: NumPy (lightweight mirror)**
1. `python IP_NUMPY/ip_numpy_train.py` iterates over the algorithms listed in its `__main__` block (defaults: TD3, DDPG, SAC, PPO, A2C) and stores `best_model.zip` files in `IP_NUMPY/ip_numpy_models/<algo>/`.
2. Reuse the shared evaluator via `python ip_eval.py --algo all --root IP_NUMPY/ip_numpy_models --episodes 5` to compare against the JAX runs.

**Workflow: MATLAB/Simulink**
1. With the MATLAB engine installed, run `python IP_MATLAB/ip_train_simulink.py --algo ppo --timesteps 100000` (adjust flags as needed). Checkpoints land in `IP_MATLAB/models/<algo>/` with TensorBoard logs in `IP_MATLAB/logs/<algo>/`.
2. Score the Simulink-trained agents using `python ip_eval.py --algo ppo --root IP_MATLAB/models --episodes 5`; this invokes the Simulink plant through `ip_simulink_env.py`.

**Workflow: PID benchmark**
- Run `python PID/PIDMainPendulum.py` or open `PID/pendSimPID.slx` inside MATLAB to reproduce the classical controller baseline.

### Running Cart-Pole

**Available Environments**
- `CP_JAX/`: JAX environment used for the final training runs and hyperparameter studies included in the report.
- `CP_NUMPY/`: NumPy-based Gym wrapper that mirrors the JAX setup for offline experiments.
- `CP_MATLAB/` + `PendCart.slx`: MATLAB/Simulink model for physics-accurate validation and classic-control comparisons.
- `PID/`: Shared PID tooling for cross-system baselines.

Each stack can be executed independently; the documented results use the JAX workflows, with MATLAB/Simulink and NumPy implementations kept for verification and ablation studies.

**Workflow: JAX (reported results)**
1. (Optional) `python cp_jax_hp.py` refreshes the Optuna studies and JSON configs in `CP_JAX/jax_hp_results/`.
2. `python cp_jax_train.py --algos all --timesteps 200000 --noise --noise-level 0.01` produces the published agents and writes checkpoints under `CP_JAX/jax-*` directories. Remove `--noise` for clean runs or narrow the set with `--algos ppo td3`.
3. Evaluate the JAX models via `python cp_eval.py --algo all --root CP_JAX/jax_clean --episodes 5`, pointing `--root` to tags like `CP_JAX/jax-41` whenever you train elsewhere.

**Workflow: NumPy (lightweight mirror)**
1. `python CP_NUMPY/cp_numpy_train.py` walks through the configured algorithms and stores outputs in `CP_NUMPY/numpy/<algo>/`.
2. Benchmark them with `python cp_eval.py --algo all --root CP_NUMPY/numpy --episodes 5 --env-noise 0.01` (drop `--env-noise` for deterministic scoring).

**Workflow: MATLAB/Simulink**
1. Execute `python CP_MATLAB/cp_train_simulink.py --algo ppo --timesteps 200000` to train directly against `PendCart.slx`. Models and logs are written to `CP_MATLAB/models/<algo>/` and `CP_MATLAB/logs/<algo>/`.
2. Evaluate Simulink agents alongside the others with `python cp_eval.py --algo ppo --root CP_MATLAB/models --episodes 5`.

**Workflow: PID benchmark**
- Launch `python PID/PIDMainCP.py` (or the accompanying MATLAB scripts) to reproduce the classical-cart baseline.

### Running the Buck Converter

**Available Environments**
- `SB3_tests/BC/BCPythonEnv.py`: NumPy-based Gym wrapper used by the Python training scripts to train the models because of the MATLAB overhead.
- `SB3_tests/BC/BCSimulinkEnv.py` + `bcSim.slx`: MATLAB/Simulink environment used for the results reported in the project.
- `PID/`: Shared PID baselines for side-by-side comparisons with the DRL controllers.

The published buck-converter performance metrics were generated with the Simulink evaluation; the NumPy scripts were used to train the models.

**Workflow: Python (NumPy)**
1. (Optional) `python hyperparameter_tuning.py` reruns the Optuna study and refreshes the JSON files stored next to each model folder.
2. `python BCPythonTrain.py` traverses the `MODELS_TO_TRAIN` and `NOISE_LEVELS` lists in its `__main__` block, writing the model zips to `models/<algo>/Seed_<seed>_Noise_<noise>/` and logging to `PY_BC_Results/<algo>/` when evaluated.
3. Evaluate the NumPy-trained agents with `python BCPythonEval.py`, which creates the plots and metrics for every noise setting.

**Workflow: MATLAB/Simulink (reported results)**
1. `python BCSimulinkTrain.py` loads the same hyperparameter files and trains each algorithm against `bcSim.slx`, saving outputs to `SIM_BC_Results/<algo>/` along with replay buffers.
2. Assess those controllers using `python BCSimulinkEval.py`, which replays the Simulink plant and exports comparison plots, saving the outputs to `SIM_BC_Results/<algo>/`.

**Workflow: PID benchmark**
- Run `python PID/PIDMainBC.py` to retrieve the results of PID for the buck converter.

**Logs and Artifacts**
- `SB3_tests/BC/models/<algo>/Seed_<seed>_Noise_<noise>/` contains `final_model.zip`, and the `hyperparameters.json` used for each run on different noise levels.
- `SB3_tests/BC/PY_BC_Results/<algo>/` stores training logs, evaluation plots, and TensorBoard traces emitted by `BCPythonTrain.py` and `BCPythonEval.py`.
- `SB3_tests/BC/buck_converter_tuning_logs/<algo>/` captures Optuna TensorBoard output from `hyperparameter_tuning.py`.
- SQLite studies named `<algo>-bc-tuning-seed42.db` are written beside the tuning script to support resuming Optuna sweeps.

### Running the Buck-boost Converter

**Available Environments**
- `SB3_tests/BBC/np_bbc_env.py`: NumPy/JAX hybrid environment used by the Python training workflow.
- `SB3_tests/BBC/BBCSimulink_env.py` + `bbcSim.slx`: MATLAB/Simulink setup that produced the experimental results in the report.
- `PID/`: Reusable PID utilities for converter baselines.

The reported buck-boost benchmarks were collected from the Simulink runs; the NumPy environment supports faster prototyping and transfer analyses.

**Workflow: Python (NumPy/JAX)**
1. (Optional) `python np_tune_bbc.py --algo sac --n-trials 50 --n-parallel 4` refreshes the Optuna sweeps and writes JSON summaries to `bbc_17_hp_results/<algo>_best_params.json`.
2. `python np_bbc_train.py --algo sac --timesteps 3000000 --noise all --device cuda` trains across the preset noise levels, storing checkpoints and VecNormalize stats under `jax_models_80/<algo>_noise_XXX/`. Use `--noise single --voltage-noise-std 0.01` to focus on one condition.
3. Evaluate those agents with `python np_eval_bbc.py --root jax_models_80 --algo sac --episodes 5 --eval-noise 0.0 0.01`, which writes metrics and plots to `eval_runs/`.

**Workflow: MATLAB/Simulink (reported results)**
1. Use MATLAB/Simulink with `bbcSim.slx` (or the Python bridge `BBCSimulink_env.py`) to train RL agents; exported models are stored in your chosen Simulink workspace directory.
2. Replay and benchmark Simulink checkpoints using `python simulink_eval_bbc.py --root <simulink_model_dir> --algo sac --episodes 5 --model-name bbcSim`, which writes results to `eval_simulink_runs_80/`.

**Workflow: PID benchmark**
- Run `python PID/PIDMainBBC.py` or open `PID/bbcSimPID.slx` in MATLAB to compare against the classical buck-boost controller.

**Logs and Artifacts**
- `SB3_tests/BBC/jax_models_80/<algo>_noise_XXX/` keeps the trained checkpoints, replay buffers, and per-run CSV logs emitted by `np_bbc_train.py`.
- `SB3_tests/BBC/jax_train_logs_110/<algo>_noise_XXX/` holds TensorBoard traces for each noise setting.
- Hyperparameter sweeps write TensorBoard data to `SB3_tests/BBC/bbc_hp_logs/<algo>/`, JSON summaries to `bbc_hp_results/<algo>_best_params.json` (copied into `bbc_17_hp_results/` for training), and SQLite studies to `bbc_jax_optuna_<algo>.db`.
- `SB3_tests/BBC/eval_runs/<ALGO>/<condition>/<run>/` is where `np_eval_bbc.py` stores evaluation metrics and plots.
- `SB3_tests/BBC/eval_simulink_runs_80/` collects the outputs from `simulink_eval_bbc.py` when benchmarking Simulink-trained agents.

## Acknowledgements

We want to express our gratitude to our supervisor, K. Prag, for her invaluable guidance and feedback throughout this research project.


  
