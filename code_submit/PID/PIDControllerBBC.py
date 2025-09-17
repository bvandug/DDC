# PIDControllerBBC.py

class PIDControllerBBC:
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
    def __init__(self, Kp, Ki, Kd, dt=None, saturation_min=0.1,
                 saturation_max=0.9):
        """Initializes the PID controller.

        Args:
            Kp (float): The proportional gain coefficient.
            Ki (float): The integral gain coefficient.
            Kd (float): The derivative gain coefficient.
            dt (float, optional): The time step. Defaults to None.
            saturation_min (float, optional): The minimum output value. Defaults to 0.1.
            saturation_max (float, optional): The maximum output value. Defaults to 0.9.
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
        """Resets the controller's internal state for a new episode."""
        self.integral_error = self.saturation_min
        self.previous_error = 0
        self.previous_time = None
        self.in_saturation = False
        print("PID controller state has been reset.")
        return True  # Return a value to confirm the method ran.

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
            voltage (float): The pre-calculated error signal, which is (Desired Voltage - Measured Voltage).
            time (float): The current simulation time.

        Returns:
            float: The saturated control output signal.
        """
        # The 'voltage' input argument is the error itself.
        error = voltage

        if self.previous_time is None:
            self.previous_time = time
            initial_output = self.Kp * error + self.Ki * self.integral_error
            return self.saturation(initial_output)

        self.dt = time - self.previous_time

        # Anti-windup: only integrate if the controller is not saturated
        # to prevent the integral term from growing uncontrollably.
        if not self.in_saturation:
            self.integral_error += error * self.dt

        derivative_error = (error - self.previous_error) / self.dt

        # Calculate the raw PID output.
        output = (self.Kp * error +
                  self.Ki * self.integral_error +
                  self.Kd * derivative_error)

        # Update state variables for the next iteration.
        self.previous_error = error
        self.previous_time = time

        saturated_output = self.saturation(output)

        return saturated_output

# Instantiate the PIDControllerBBC object.
pid_controller = PIDControllerBBC(Kp=1.3, Ki=0, Kd=0.001)

def controller_call(voltage, time):
    """A wrapper function to call the PID controller's update method.

    Args:
        voltage (float): The pre-calculated error signal.
        time (float): The current simulation time.

    Returns:
        float: The computed control signal from the PID controller.
    """
    signal = pid_controller.update(voltage, time)
    return signal


def reset_controller():
    """A wrapper function to allow external calls to the reset method.

    This is useful for interfacing with applications like MATLAB, providing a
    simple entry point to reset the controller state.

    Returns:
        bool: True if the controller was successfully reset.
    """
    return pid_controller.reset()