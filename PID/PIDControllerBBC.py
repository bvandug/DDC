class PIDControllerBBC:
    '''
    A class that implements a Proportional-Integral-Derivative (PID) controller
    with saturation and anti-windup measures to the BBC.
    
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

    def update(self, goal, v_out, time):
        '''
        Computes the control output using the PID control equation.

        Args:
            goal (float): Desired output voltage.
            v_out (float): Observed BBC output voltage.
            time (float): Current time.

        Returns:
            float: Saturated output signal.
        '''
        
        error = v_out-goal

        # return intial duty cycle
        if self.previous_time is None:
            self.previous_time = time
            initial_output = self.Kp * error + self.Ki * self.integral_error
            return self.saturation(initial_output)

        self.dt = time - self.previous_time

        # anti-windup measures included in calculation of integral error
        if not (self.in_saturation):
            self.integral_error += error * self.dt

        derivative_error = (error - self.previous_error) / self.dt

        # Calculate the PID output
        output = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)

        self.previous_error = error
        self.previous_time = time

        saturated_output = self.saturation(output)
        
        return saturated_output

# Instantiate PIDControllerBBC object.
pid_controller = PIDControllerBBC(Kp=1.3, Ki=0, Kd=0.001)

def controller_call(goal, v_out, time):
    '''
    Calls the PID controller to compute the control signal.

    Args:
        v_out (float): Observed BBC output voltage.
        time (float): Current time.

    Returns:
        float: Saturated output signal.
    '''
    signal = pid_controller.update(goal, v_out, time)
    return signal