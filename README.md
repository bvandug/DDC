# Data-Driven Control

This project compares the performance of data-driven control methods, Q-Learning and NEAT, with one another, as well as traditional PID control. The methods are applied to non-linear binary control systems, specifically the inverted pendulum and DC-DC buck converters.

## Authors
**Matthew Fleischman**<br>
*University of Cape Town* <br>
*FLSMAT002@myuct.ac.za* 
<br>

**Ariel J. Levy**<br>
*University of Cape Town* <br>
*LVYARI002@myuct.ac.za*
___

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Installation

1. Ensure that you are using Python version 3.9, 3.10, or 3.11. For Windows users, Python must be installed via the Python website, not the Microsoft store.

2. Install MATLAB on your system. It can be downloaded from [MathWorks](https://www.mathworks.com/products/matlab.html).

3. Clone this repository:
    ```bash
   git clone https://github.com/AJ-Levy/Data-Driven-Control.git
    ```

4. Place this directory within the MATLAB directory:
   ```bash
   mv Data-Driven-Control MATLAB/
   ```
   
5. Navigate to the project directory:
    ```bash
    cd MATLAB/Data-Driven-Control
    ```
    
6. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

7. Launch MATLAB and verify that the correct Python interpreter is being used with the following commands:
    ```MATLAB
    python_interpretor = '<path_to_python_version>';
    pyenv('Version', python_interpretor);
    ```
    
## Usage

To run the simulations for each control method on the respective systems, follow these steps:

### 1. Ensure All Dependencies Are Installed
Make sure the necessary Python packages are installed as per the installation instructions.

### 2. Choose the Desired Control Method and System
Each control method has a main Python file for every system to which it was applied. Below is a list of the main files corresponding to each system and method. 

 - Inverted Pendulum

    - PID: Run `PIDMainPend.py`
    - Q-Learning: Run `QLearningMainPend.py`
    - NEAT: Run `NEATMainPend.py` 

- Buck Converter

    - PID: Run `PIDMainBC.py`
    - Q-Learning: Run `QLearningMainBC.py`

- Buck-Boost Converter

    - PID: Run `PIDMainBBC.py`
    - NEAT: Run `NEATMainBBC.py`

### 3. Execution
Each main file is self-contained. Running the script will automatically start the simulation, displaying results like control method details and plots of system behaviour. 

First navigate to the correct directory: `PID`, `QLearning`, or `NEAT`, using the following command:
```bash
cd <control_method>
```

To execute a file, use the command line or your preferred Python environment:
```bash
python <control_method>Main<system>.py
```

### 4. Adjustments

To modify the parameters of a specific control method, follow these instructions:

- PID

    Go to the appropriate `PIDController` file and change the values of `Kp`, `Ki`, and `Kd` in this line of code:
    ```python
    pid_controller = PIDController(Kp=x, Ki=y, Kd=z)
    ```

- Q-Learning

    Go to the appropriate `QLearningAgent` file and change the values of `alpha` and `gamma` in this line of code.
    ```python
    agent = QLearningAgent(alpha=s, gamma=t) 
    ```

- NEAT
  
    1. Go to the appropriate `Config` file and adjust the configuarion parameters of the NEAT population. It is advised to not          change the fitness criterion or number of hidden, input or output nodes. Note that NEAT uses custom activation functions         for the inverted pendulum.

    2. To change the number of generations, go to the appropriate `NEATMain` file and change the last argument in this function.
       ```python
       pop.run(eval_genomes, 25)
       ```

    3. In the context of the BBC, to change the desired voltage for the NEAT population that you are training, go to 
       `NEATMainBBC` and adjust the `final_goal` variable in the `eval_genomes(genomes, config)` method. 

Adjust these parameters according to your needs to fine-tune the performance of the control methods.

## Acknowledgements

We would like to express our gratitude to our supervisor, Prof. K. Prag, for her invaluable guidance and feedback throughout this research project.

  
