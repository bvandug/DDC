import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

def setNoise(eng, model, noise):
    '''
    Sets amount of noise to be supplied to the state variables.

    Args:
        eng (MatlabEngine Object): The instance of Matlab currently running.
        model (str): Name of the Simulink model in use.
        noise (bool): Whether noise should be supplied or not.
    '''
    if noise:
        eng.set_param(f'{model}/Noise', 'Cov', str([0.000006]), nargout=0)
        random_seed = np.random.randint(1, 100000)
        eng.set_param(f'{model}/Noise', 'seed', str([random_seed]), nargout=0)
    else:
        eng.set_param(f'{model}/Noise', 'Cov', str([0]), nargout=0)

def main(noise = False,
         model = 'pendSimPID',
         mask = 'Pendulum and Cart',
         stabilisation_precision = 0.05,
         initial_angle = np.pi/3):
    '''
    Method to set up MATLAB, Simulink, and handle data aquisition/plotting.

    Args:
        noise (bool): Whether noise should be supplied or not.
        model (str): Name of the Simulink model in use.
        mask (str): Name of the model block in use.
        stabilisation_precision (float): Magnitude of error bounds around 0 rad.
        initial_angle (float): Angle (in radians) from which the pendulum stabilises.
    '''
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)

    # Setting model parameters
    eng.set_param(f'{model}/{mask}', 'init', str(initial_angle), nargout=0)
    setNoise(eng, model, noise)

    print("Running simulation...")
    eng.eval(f"out = sim('{model}');", nargout=0)
    print("Simulation complete")

    # Get angles
    angle_2d = eng.eval("out.angle")
    angle_lst = []
    for angle in angle_2d:
        angle_lst.append(angle[0])

    # Get time
    time_2d = eng.eval("out.tout")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0])

    eng.quit()

    # Plotting acquired data
    plt.plot(time_lst, angle_lst, label = f"{round(initial_angle,2)} init. angle")
    plt.axhline(y=stabilisation_precision, color='k', linestyle='--', label=f'{stabilisation_precision} rad')
    plt.axhline(y=-stabilisation_precision, color='k', linestyle='--', label=f'-{stabilisation_precision} rad')
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.title("Angle of pendulum over time")
    plt.xlim(0,max(time_lst))
    plt.ylim(-np.pi/2,np.pi/2)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main(noise=False)