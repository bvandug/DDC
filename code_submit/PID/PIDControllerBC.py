# PIDControllerBC.py

class PIDController:
    """A PID controller with saturation and anti-windup.

    Attributes:
        Kp (float): The proportional gain.
        Ki (float): The integral gain.
        Kd (float): The derivative gain.
        dt (float): The time step between updates.
        integral_error (float): The accumulated integral error.
        previous_error (float): The error from the previous time step.
        previous_time (float): The timestamp of the previous update.
        saturation_min (float): The lower output limit of the controller.
        saturation_max (float): The upper output limit of the controller.
        in_saturation (bool): Flag indicating if the output is saturated.
    """
    def __init__(self, Kp, Ki, Kd, dt=None, saturation_min=0.1, saturation_max=0.9):
        """Initializes the PID controller.

        Args:
            Kp (float): The proportional gain coefficient.
            Ki (float): The integral gain coefficient.
            Kd (float): The derivative gain coefficient.
            dt (float, optional): The time step. Defaults to None.
            saturation_min (float, optional): The minimum output value.
            saturation_max (float, optional): The maximum output value.
        """
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
        """Resets the controller's internal state for a new episode.

        Returns:
            bool: True to confirm the method ran successfully.
        """
        self.integral_error = self.saturation_min
        self.previous_error = 0
        self.previous_time = None
        self.in_saturation = False
        print("  PID controller state has been reset.")
        return True

    def saturation(self, signal):
        """Clips the output signal to stay within the saturation limits.

        Args:
            signal (float): The raw controller output signal.

        Returns:
            float: The signal clipped between saturation_min and saturation_max.
        """
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
        """Computes the control output using the PID control equation.

        Args:
            voltage (float): The voltage error signal (Desired - Measured).
            time (float): The current simulation time.

        Returns:
            float: The saturated control output signal.
        """
        error = voltage
        if self.previous_time is None:
            self.previous_time = time
            initial_output = self.Kp * error + self.Ki * self.integral_error
            return self.saturation(initial_output)
        self.dt = time - self.previous_time
        if not (self.in_saturation):
            self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        output = (self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative_error)
        self.previous_error = error
        self.previous_time = time
        saturated_output = self.saturation(output)
        return saturated_output

pid_controller = PIDController(Kp=0.45, Ki=43.5, Kd=0)

def controller_call(voltage, time):
    """A wrapper function to call the PID controller's update method.

    Args:
        voltage (float): The voltage error signal.
        time (float): The current simulation time.

    Returns:
        float: The calculated control signal from the PID controller.
    """
    signal = pid_controller.update(voltage, time)
    return signal

def reset_controller():
    """A wrapper function to allow external calls to the reset method.

    Returns:
        bool: True if the controller was successfully reset.
    """
    return pid_controller.reset()