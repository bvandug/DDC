This repository contains a collection of Python scripts for simulating and evaluating PID controllers for various systems using the MATLAB engine. The included systems are: a Buck Converter, a Boost-Buck Converter, a Cart-Pole, and an Inverted Pendulum.

## Table of Contents

  - [Overview](https://www.google.com/search?q=%23overview)
  - [Requirements](https://www.google.com/search?q=%23requirements)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [Systems](https://www.google.com/search?q=%23systems)
      - [Buck Converter](https://www.google.com/search?q=%23buck-converter)
      - [Boost-Buck Converter](https://www.google.com/search?q=%23boost-buck-converter)
      - [Cart-Pole](https://www.google.com/search?q=%23cart-pole)
      - [Inverted Pendulum](https://www.google.com/search?q=%23inverted-pendulum)
  - [Controllers](https://www.google.com/search?q=%23controllers)
  - [License](https://www.google.com/search?q=%23license)

## Overview

This project provides a framework for running simulations of different physical and electrical systems controlled by PID controllers. The simulations are performed in MATLAB/Simulink, while the main simulation loops, data processing, and plotting are handled by Python scripts. The primary goal is to evaluate the performance of the PID controllers under various conditions, such as different noise levels and initial states.

## Requirements

To run the simulations, you will need the following software installed:

  - Python 3.x
  - MATLAB with Simulink
  - MATLAB Engine API for Python
  - NumPy
  - Matplotlib

You can install the required Python libraries using pip:

```bash
pip install numpy matplotlib
```

For information on setting up the MATLAB Engine API for Python, please refer to the official MathWorks documentation.

## Usage

Each system has a main script that runs the simulation and generates performance metrics and plots. To run a simulation, execute the corresponding `PIDMain...py` script from your terminal. For example, to run the Buck Converter simulation:

```bash
python PIDMainBC.py
```

The scripts will automatically start the MATLAB engine, run the Simulink models, and output the results to the console and as `.svg` plot files.

## Systems

This repository includes simulations for four different systems:

### Buck Converter

  * **Description**: A step-down DC-DC converter.
  * **Main Script**: `PIDMainBC.py`
  * **Controller**: `PIDControllerBC.py`
  * **Simulink Model**: `bcSimPID.slx`
  * **Functionality**: The script `PIDMainBC.py` simulates a buck converter, aiming to achieve a desired output voltage from a higher source voltage. It tests the PID controller under different noise levels and calculates metrics such as stabilization time, overshoot, and steady-state error. The PID controller in `PIDControllerBC.py` includes features like saturation and anti-windup.

### Boost-Buck Converter

  * **Description**: A DC-DC converter that can step-up or step-down voltage.
  * **Main Script**: `PIDMainBBC.py`
  * **Controller**: `PIDControllerBBC.py`
  * **Simulink Model**: `bbcSimPID.slx`
  * **Functionality**: `PIDMainBBC.py` evaluates the performance of a PID controller for a boost-buck converter across various noise levels and target voltages. It generates plots of the voltage response and calculates average performance metrics over multiple simulation runs. The controller in `PIDControllerBBC.py` also features saturation and anti-windup mechanisms.

### Cart-Pole

  * **Description**: A classic control theory problem where the goal is to balance a pole on a moving cart.
  * **Main Script**: `PIDMainCP.py`
  * **Controller**: `PIDControllerCP.py`
  * **Simulink Model**: `pendSimPID.slx`
  * **Functionality**: The `PIDMainCP.py` script runs simulations for the cart-pole system, testing the PID controller's ability to stabilize the pole from various initial angles and under different noise conditions. It calculates and displays performance metrics in a summary table. The PID controller in `PIDControllerCP.py` has an optional output clamping feature.

### Inverted Pendulum

  * **Description**: A system where a pendulum is balanced upright.
  * **Main Script**: `PIDMainPendulum.py`
  * **Controller**: `PIDControllerPendulum.py`
  * **Simulink Model**: `pendulumSimPID.slx`
  * **Functionality**: `PIDMainPendulum.py` simulates an inverted pendulum system, evaluating the PID controller's performance in stabilizing the pendulum from different starting angles and with varying levels of observational noise. The script generates individual and summary plots, and a detailed performance table. The controller in `PIDControllerPendulum.py` includes action clipping to limit the control force.

## Controllers

Each system has a dedicated PID controller script:

  - `PIDControllerBC.py`: For the Buck Converter, with saturation and anti-windup.
  - `PIDControllerBBC.py`: For the Boost-Buck Converter, also with saturation and anti-windup.
  - `PIDControllerCP.py`: For the Cart-Pole, with optional output clamping.
  - `PIDControllerPendulum.py`: For the Inverted Pendulum, with action clipping.

Each controller script defines a PID controller class and wrapper functions to be called from the main simulation scripts and Simulink.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
