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

### Running the Inverted Pendulum

### Running Cart-Pole

### Running the Buck Converter

#### Running PID



#### Running DRL

**Running the Models**
1. Run hyperparameter tuning for the DRL algorithms (optional) by running ```python hyperparameter_tuning.py``` <br>
  This will run an Optuna study to find the best hyperparameters for the different DRL algorithms and save the results. <br>

**Training the Models**
1. To train with the Python environment, run: ```python BCPythonTrain.py``` <br>
   This script will load in a specific set of hyperparameters. <br>
2. To train with the Simulink environment, run: ```python BCSimulinkTrain.py```<br>
   This will train the DRL agent using the Simulink model as the environment. It also loads a specific set of hyperparameters (Note: the simulink training has significant overhead)<br>

**Evaluating the Models**
After training, you can evaluate the performance of the saved models.<br>
1. To evaluate the models in a Python environment, run: ```python BCPythonEval.py```<br>
2. To evaluate the models in the Simulink environment, run: ```python BCSimulinkEval.py```<br>
These scripts will run the models through a series of evaluation episodes, calculate performance metrics, and save plots of the results.<br>
   


### Running the Buck-boost Converter


## Running the Simulation


## Acknowledgements

We would like to express our gratitude to our supervisor, K. Prag, for her invaluable guidance and feedback throughout this research project.

[text](https://www.ijert.org/research/design-and-analysis-of-buck-converter-IJERTV3IS031844.pdf)

  
