import dill
import os

class NEATControllerPend:
    '''
    Class to store ANN.
    '''
    def __init__(self):
        self.net = None

net_obj = NEATControllerPend()

def controller_call(theta, theta_v, time):
    '''
    Method that MATLAB calls for NEAT.
    '''    
    # At start get new ANN.
    if time == 0.0:
        with open('network.dill', 'rb') as f:
            net_obj.net = dill.load(f)
        os.remove("network.dill")

    # Get output force
    inputs = [theta, theta_v]
    force = (net_obj.net.activate(inputs)[0])
    return force