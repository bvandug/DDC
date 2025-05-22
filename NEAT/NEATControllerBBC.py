import dill
import os

class NEATcontrollerBBC:
    '''
    Class to store ANN and previous error.
    '''
    def __init__(self):
        self.prev_error = 0
        self.net = None

net_obj = NEATcontrollerBBC()

def controller_call(goal, v_out, time):
    '''
    Method that MATLAB calls for NEAT.
    '''    
    # At start get new ANN and reset error
    if time == 0.0:
        with open('network.dill', 'rb') as f:
            net_obj.net = dill.load(f)
        os.remove("network.dill")
        net_obj.prev_error = 0

    # Error calculations
    error = (v_out-goal)
    derivative_error = (error - net_obj.prev_error) / 5e-6

    # Get duty cycle and update previous error
    inputs = [v_out, error, derivative_error]
    duty = (net_obj.net.activate(inputs)[0])
    net_obj.prev_error = error

    # Saturation
    if goal > -48:
        if duty >= 0.5:
            duty = 0.5
        elif duty <= 0.1:
            duty = 0.1
    else:
        if duty >= 0.9:
            duty = 0.9
        elif duty <= 0.5:
            duty = 0.5

    return duty