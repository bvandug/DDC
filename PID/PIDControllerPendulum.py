# PIDControllerPendulum.py

class PIDController:
    """A PID controller for a setpoint-zero system with action clipping."""
    def __init__(self, Kp, Ki, Kd, dt):
        """Initializes the PID controller for the Inverted Pendulum.

        Args:
            Kp (float): The proportional gain coefficient.
            Ki (float): The integral gain coefficient.
            Kd (float): The derivative gain coefficient.
            dt (float): The time step of the simulation.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = 0
        self.previous_error = 0
        self.enable_clipping = False
        self.action_limit = float('inf')

    def reset(self):
        """Resets the controller's internal state for a new simulation run.

        Returns:
            bool: True to confirm the method ran successfully.
        """
        self.integral_error = 0
        self.previous_error = 0
        print("  PID controller state has been reset.")
        return True

    def set_clipping_params(self, enabled, limit):
        """Sets the parameters for action clipping for the next run.

        Args:
            enabled (bool): The desired clipping state (True or False).
            limit (float): The absolute maximum value for the output force.

        Returns:
            None
        """
        self.enable_clipping = enabled
        self.action_limit = abs(limit) if enabled else float('inf')

    def update(self, measurement):
        """Calculates the control output based on the current measurement.

        Args:
            measurement (float): The current angle measurement from the system.

        Returns:
            float: The raw (unclipped) control force.
        """
        error = 0 - measurement
        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        output_force = (self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative_error)
        self.previous_error = error
        return output_force

pid_instance = PIDController(Kp=2, Ki=0, Kd=0.3, dt=0.001)

def controller_call(measurement):
    """Function called by Simulink to get the control signal.

    Args:
        measurement (float): The current angle measurement from the system.

    Returns:
        float: The final control force (potentially clipped).
    """
    raw_force = pid_instance.update(measurement)
    if pid_instance.enable_clipping:
        clipped_force = max(min(raw_force, pid_instance.action_limit), -pid_instance.action_limit)
        return clipped_force
    else:
        return raw_force

def reset_controller():
    """Wrapper function for Simulink to reset the controller state.

    Returns:
        bool: True if the controller was successfully reset.
    """
    return pid_instance.reset()

def configure_controller_clipping(enabled, limit):
    """Wrapper function to configure clipping from an external script.

    Args:
        enabled (bool): The desired clipping state (True or False).
        limit (float): The absolute maximum value for the output force.

    Returns:
        bool: True to confirm the method ran successfully.
    """
    pid_instance.set_clipping_params(enabled, limit)
    return True