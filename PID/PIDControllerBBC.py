# Updated PIDControllerBBC.py

class PIDControllerBBC:
    '''
    A class that implements a Proportional-Integral-Derivative (PID) controller
    with saturation and anti-windup measures to the BBC. This version is
    standardized to match the BC controller implementation.
    '''
    def __init__(self, Kp, Ki, Kd, dt=None, saturation_min = 0.1, saturation_max = 0.9):
        '''
        Initialises the PID controller with the specified gain coefficients, step size, and saturation thresholds.
        '''
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = saturation_min
        self.previous_error = 0
        self.previous_time = None
        self.saturation_min = saturation_min
        self.saturation_max = saturation_max
        self.in_saturation = False

    def reset(self):
        """Resets the controller's internal state for a new episode."""
        self.integral_error = self.saturation_min
        self.previous_error = 0
        self.previous_time = None
        self.in_saturation = False
        print("  PID controller state has been reset.")
        return True # Return a value to confirm it ran

    def saturation(self, signal):
        '''
        Ensures the output remains between the saturation limits.
        '''
        if signal > self.saturation_max:
            self.in_saturation = True
            return self.saturation_max
        elif signal < self.saturation_min:
            self.in_saturation = True
            return self.saturation_min
        else:
            self.in_saturation = False
        return signal

    def update(self, voltage, time):
        '''
        Computes the control output using the PID control equation.
        The 'voltage' argument is now the pre-calculated error signal from Simulink.

        Args:
            voltage (float): Voltage error signal (Desired Voltage - Measured Voltage).
            time (float): Current time.
        '''
        error = voltage # The input is now the error itself [cite: 171]

        if self.previous_time is None:
            self.previous_time = time
            initial_output = self.Kp * error + self.Ki * self.integral_error
            return self.saturation(initial_output)

        self.dt = time - self.previous_time

        # Anti-windup: only integrate if the controller is not saturated [cite: 170]
        if not (self.in_saturation):
            self.integral_error += error * self.dt

        derivative_error = (error - self.previous_error) / self.dt

        output = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)
        
        self.previous_error = error
        self.previous_time = time

        saturated_output = self.saturation(output)
        
        return saturated_output

# Instantiate PIDControllerBBC object.
# Note: The gains are specific to the BBC's tuning and are kept the same.
pid_controller = PIDControllerBBC(Kp=1.3, Ki=0, Kd=0.001)

def controller_call(voltage, time):
    '''
    Calls the PID controller to compute the control signal.
    '''
    signal = pid_controller.update(voltage, time)
    return signal

def reset_controller():
    """Allows MATLAB to call the reset method, similar to PIDControllerBC.py."""
    return pid_controller.reset()