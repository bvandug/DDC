# Evaluating DRL for control systems
The project focuses on evaluating:
- The Stablebaselines3 DRL algorithms (PPO, A2C, DQN, DDPG, TD3, SAC) on various control systems.
- Creating simulation environments that will train SB3 models faster but with similar fidelity

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

# Full code base:
- The full code base is available at: https://github.com/bvandug/DDC
- This includes tuned models, evals, databases and tuning and hyperparameter logs.

# Structure for DRL Methods Explained

To explore the current DRL work, browse the `SB3_tests` folder in this repository. 
The PID benchmark work is done in `PID`.

## Systems Included

- **BC**: Buck Converter  
- **BBC**: Buck-Boost Converter  
- **Inverted Pendulum**
- **Cartpole**


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

### Running the Inverted Pendulum

**Running the Models**
1. (Optional) From `SB3_tests/Inverted_Pendulum/ip_jax` run `python ip_jax_hp.py` to regenerate Optuna tuning results. Adjust the algorithm list at the bottom of the script when you want to limit the sweep.

**Training the Models**
1. `python ip_jax_train.py --algos all --timesteps 100000` trains every configured SB3 agent on the accelerated JAX environment and stores checkpoints in `ip_jax/jax_models/<algo>_noise_XXX`. Add `--algos ppo sac` to focus on a subset or `--noise --noise-level 0.01` to include observation noise.
2. (Optional) `python IP_NUMPY/ip_numpy_train.py` loops over the algorithms listed in its `__main__` block (defaults: TD3, DDPG, SAC, PPO, A2C) and saves `best_model.zip` files under `ip_numpy_models/<algo>/`.

**Evaluating the Models**
1. From `SB3_tests/Inverted_Pendulum` run `python ip_eval.py --algo all --root ip_jax/jax_models --episodes 5` to score the JAX-trained checkpoints. Use flags like `--env-noise 0.01` or `--name-contains noise_0.010` to filter runs.
2. Point the same command at `--root ip_numpy_models` when you want to evaluate the NumPy-trained agents.

### Running Cart-Pole

**Running the Models**
1. (Optional) From `SB3_tests/Cartpole/CP_JAX` run `python cp_jax_hp.py` to refresh the Optuna studies. Edit the algorithm list at the bottom of the script to control which agents are tuned.

**Training the Models**
1. `python cp_jax_train.py --algos all --timesteps 200000 --noise --noise-level 0.01` trains every configured algorithm on the JAX environment and saves checkpoints under the `CP_JAX/jax-*` folders. Drop the `--noise` flag for clean runs or pass `--algos ppo td3` to restrict the set.
2. (Optional) `python CP_NUMPY/cp_numpy_train.py` iterates over the algorithms listed in its `__main__` block and writes NumPy-based checkpoints to `CP_NUMPY/numpy/<algo>/`.

**Evaluating the Models**
1. From `SB3_tests/Cartpole` run `python cp_eval.py --algo all --root CP_JAX/jax_clean --episodes 5` to evaluate the JAX-trained agents; retarget `--root` to directories like `CP_JAX/jax-41` when you train elsewhere.
2. Use the same command with `--root CP_NUMPY/numpy` to score the NumPy-trained models, and add `--env-noise 0.01` to test robustness at different noise levels.

*Optional orchestration*: `python jax_pipeline.py` automates the train/eval cycle for the configured algorithms and noise settings.

### Running the Buck Converter

#### Running PID
- The PID baselines in `PID/` are optional and only needed when you want classical control comparisons.

#### Running DRL
**Running the Models**
1. (Optional) Launch the Optuna sweeps by running `python hyperparameter_tuning.py`. This script will search the Stable-Baselines3 hyperparameter space and write the best configurations alongside each model folder.

**Training the Models**
1. To train with the Python environment, run `python BCPythonTrain.py`. The script reads the saved hyperparameters, cycles through the algorithms and noise levels configured in its `__main__` block, and logs outputs to `models/` and `PY_BC_Results/`.
2. To train with the Simulink environment, run `python BCSimulinkTrain.py`. This uses the same hyperparameter files but steps the Simulink plant; note that these runs have higher overhead and save results under `SIM_BC_Results/`.

**Evaluating the Models**
After training, you can evaluate the performance of the saved policies.
1. To evaluate the Python-environment checkpoints, run `python BCPythonEval.py`.
2. To evaluate the Simulink checkpoints, run `python BCSimulinkEval.py`. These scripts replay the stored agents, compute metrics, and export plots for each noise setting.

*Optional cross-check*: `python compare_np_vs_simulink.py` plots numpy and Simulink rollouts side-by-side for a selected checkpoint.

### Running the Buck-boost Converter

**Running the Models**
1. (Optional) `python np_tune_bbc.py --algo sac --n-trials 50 --n-parallel 4` refreshes the Optuna studies and writes best parameters to `bbc_17_hp_results/<algo>_best_params.json`.

**Training the Models**
1. `python np_bbc_train.py --algo sac --timesteps 3000000 --noise all --device cuda` trains across the preset noise levels and stores checkpoints in `jax_models_80/<algo>_noise_XXX/`. Switch to `--noise single --voltage-noise-std 0.01` when you only need one condition.

**Evaluating the Models**
1. `python np_eval_bbc.py --root jax_models_80 --algo sac --episodes 5 --eval-noise 0.0 0.01` loads each `best_model.zip` with its VecNormalize stats and produces metrics and plots under `eval_runs/`.
2. `python simulink_eval_bbc.py --root jax_models_80 --algo sac --episodes 5` exercises the same checkpoints against the Simulink plant. Provide `--model-path` and `--stats-path` to evaluate a single run.

*Optional comparison*: `python compare_np_vs_simulink.py` overlays numpy and Simulink trajectories for a trained checkpoint.

## Running the Simulation


## Acknowledgements

We would like to express our gratitude to our supervisor, K. Prag, for her invaluable guidance and feedback throughout this research project.

[text](https://www.ijert.org/research/design-and-analysis-of-buck-converter-IJERTV3IS031844.pdf)

  
