import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    '''
    Collects and returns the workspace data.
    '''
    # Get voltages
    voltage_2d = eng.eval("out.voltage")
    voltage_lst = []
    for voltage in voltage_2d:
        voltage_lst.append(voltage[0])

    # Get time
    time_2d = eng.eval("out.tout")
    time_lst = []
    for time in time_2d:
        time_lst.append(time[0]) 

    pulse_2d = eng.eval("out.pulse")
    pulse_lst = []
    for pulse in pulse_2d:
        pulse_lst.append(pulse[0])
    
    pwm_2d = eng.eval("out.pwm")
    pwm_lst = []
    for pwm in pwm_2d:
        pwm_lst.append(pwm[0])

    return voltage_lst, time_lst, pulse_lst, pwm_lst

def main(noise_power = 0,
         goal_lst = [-30, -80, -110],
         model = 'bbcSimPID'):
    '''
    Method to set up MATLAB, Simulink, and handle data aquisition/plotting.

    Args:
        noise_power (float) : Determined the amount of noise applied.
        goal_lst (List[foat]) : The array of desired output voltages.
        model (str) : Name of the Simulink model in use.
    '''
    print("Setting up engine...")
    global eng
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)

    for goal in goal_lst:
    
        # Setting model parameters
        eng.set_param(f'{model}/Constant', 'Value', str(goal), nargout=0)
        noise_seed = str(np.random.randint(1, high=40000))
        eng.set_param(f'{model}/Noise', 'seed', f'[{noise_seed}]', nargout=0)
        eng.set_param(f'{model}/Noise', 'Cov', f'[{noise_power}]', nargout=0)

        print(f"Running simulation for {goal}V...")
        eng.eval(f"out = sim('{model}');", nargout=0)
        voltage_lst, time_lst, pulse_lst, pwm_lst = get_data()
        print("Simulation complete")

        # Get stability time
        error_bound = 0.04*goal
        stab_time = None
        for j in range(len(voltage_lst)):
            if (goal+(error_bound)) <= voltage_lst[j] <= (goal-(error_bound)):
                if stab_time == None:
                    stab_time = time_lst[j]
            else:
                stab_time = None
        if stab_time != None:
            print(f"Stab. time for {goal}V:", round(stab_time,3))
        else:
            print(f"S.tab time for {goal}V: failed")

        # Plotting acquired data
        plt.plot(time_lst, voltage_lst, label = f'Goal voltage: {goal}V')
        if goal == -110:
            plt.axhline(y=goal+error_bound, color='k', linestyle='--', label='$\pm$ 4% error bar')
        else: 
            plt.axhline(y=goal+error_bound, color='k', linestyle='--')
        plt.axhline(y=goal-error_bound, color='k', linestyle='--')

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.xlim(0,0.07)
    plt.ylim(-120,8)
    plt.legend()
    plt.show()

    eng.quit()

if __name__ == '__main__':
    main()