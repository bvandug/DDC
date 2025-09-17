# Evaluating DRL for control systems

This project builds on another comparing the performance of data-driven control methods, Q-Learning and NEAT, with one another, as well as traditional PID control.

The project focuses on evaluating:
-The Stablebaselines3 DRL algorithms (PPO, A2C, DQN, DDPG, TD3, SAC) on control systems.
-Creating simulation environments that will train SB3 models faster but with similar fidelity

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

# Structure for DRL Methods Explained

To explore the current DRL work, browse the `SB3_tests` folder in this repository. The PID benchmark work is done in PID.

## Systems Included

- **BC**: Buck Converter  
- **BBC**: Buck-Boost Converter  
- **Inverted Pendulum**
- **Cartpole**

## MATLAB Setup

To get MATLAB dependency installed:
- Get main folder path in your computer
- Run PowerShell in admin mode
- Navigate to where main folder is
- Activate virtual environment
- Enter the following commands:
   - PS> cd "C:\Program Files\MATLAB\R2024b\extern\engines\python"- where your MATLAB is installed
   - PS> & "C:\... \MATLAB\venv\Scripts\python.exe" setup.py install 


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


## Acknowledgements

We would like to express our gratitude to our supervisor, K. Prag, for her invaluable guidance and feedback throughout this research project.

[text](https://www.ijert.org/research/design-and-analysis-of-buck-converter-IJERTV3IS031844.pdf)

  
