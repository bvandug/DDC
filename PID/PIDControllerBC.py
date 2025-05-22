class PIDController:
    '''
    A class that implements a Proportional-Integral-Derivative (PID) controller
    with saturation and anti-windup measures to the BC.
    
    Attributes:
        Kp (float): Proportional gain coefficient.
        Ki (float): Integral gain coefficient.
        Kd (float): Derivative gain coefficient.
        dt (float): Step size.
        integral_error (float): Accumulated integral of the error.
        previous_error (float): Error value from the previous update.
        previous_time (float): Time from previous update.
        saturation_min (float): Minimum saturation limit.
        saturation_max (float): Maximum saturation limit.
        in_saturation (bool): Whether the output signal is being saturated.
    '''

    def __init__(self, Kp, Ki, Kd, dt=None, saturation_min = 0.1, saturation_max = 0.9):
        '''
        Initialises the PID controller with the specified gain coefficients, step size, and saturation thresholds.
    
        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.
            dt (float): Step size.
            saturation_min (float): Minimum saturation limit.
            saturation_max (float): Maximum saturation limit.
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

    def saturation(self, signal):
        '''
        Ensures the output remains between the saturation limits.

        Args:
            signal (float): Unsaturated input signal.

        Returns:
            float: Saturated output signal.
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

        Args:
            voltage (float): Voltage error signal.
            time (float): Current time.

        Returns:
            float: Saturated output signal.
        '''
        error = voltage

        # returning initial duty cycle
        if self.previous_time is None:
            self.previous_time = time
            initial_output = self.Kp * error + self.Ki * self.integral_error
            return self.saturation(initial_output)

        self.dt = time - self.previous_time

        # anti-windup measures included in calculation of integral error
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

# Instantiate PIDController object.
pid_controller = PIDController(Kp=0.45, Ki=43.5, Kd=0)

def controller_call(voltage, time):
    '''
    Calls the PID controller to compute the control signal.

    Args:
        voltage (float): Voltage error signal.
        time (float): Current time.

    Returns:
        float: Saturated output signal.
    '''
    signal = pid_controller.update(voltage, time)
    return signal
