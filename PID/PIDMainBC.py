import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

def setNoise(eng, model, noise):
    '''
    Sets amount of noise to be supplied to the state variables.

    Args:
        eng (MatlabEngine Object): The instance of Matlab currently running.
        model (str): Name of the Simulink model in use.
        noise (bool): Whether noise should be supplied or not.
    '''
    if noise:
        eng.set_param(f'{model}/Noise', 'Cov', str([0.0000008]), nargout=0)
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise', 'seed', str([random_seed]), nargout=0)
    else:
        eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)

def main(noise = False,
         desired_voltage = 30.0,
         source_voltage = 48.0,
         model = 'bcSimPID',
         stabilisation_precision = 0.5):
    '''
    Method to set up MATLAB, Simulink, and handle data aquisition/plotting.

    Args:
        noise (bool): Whether noise should be supplied or not.
        desired_voltage (float): Desired reference voltage.
        source_voltage (float): Source/Input voltage.
        model (str): Name of the Simulink model in use.
        stabilisation_precision (float): Magnitude of error bounds around the reference voltage.
    '''
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)
    
    # Setting model parameters
    setNoise(eng, model, noise)
    eng.set_param(f'{model}/finalVoltage', 'Value', str(desired_voltage), nargout=0)
    eng.set_param(f'{model}/input_voltage', 'Amplitude', str(source_voltage), nargout=0)

    print("Running simulation...")
    eng.eval(f"out = sim('{model}');", nargout=0)
    print("Simulation complete")

    # Get voltages
    voltage_2d = eng.eval("voltage")
    voltage_lst = []
    for v in voltage_2d:
        voltage_lst.append(v[0])

    # Get times
    time_2d = eng.eval("time")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0])

    eng.quit()

    # Plotting acquired data
    plt.plot(time_lst, voltage_lst, label = f'{desired_voltage} V')
    plt.axhline(y=stabilisation_precision + (desired_voltage), color='k', linestyle='--', label=f'{stabilisation_precision + (desired_voltage)} V')
    plt.axhline(y=-stabilisation_precision + (desired_voltage), color='k', linestyle='--', label=f'{-stabilisation_precision + (desired_voltage)} V')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.xlim(0,max(time_lst))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main(noise=False)