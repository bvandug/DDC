# PIDControllerCP.py

import numpy as np

class PIDControllerCP:
    """A PID controller with optional output clamping."""
    def __init__(self, Kp, Ki, Kd, dt):
        """Initializes the PID controller for the Cart-Pole system.

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
        self.max_force = 10
        self.clamping_enabled = True

    def reset(self):
        """Resets the controller's internal state for a new episode.

        Returns:
            bool: True to confirm the method ran successfully.
        """
        self.integral_error = 0
        self.previous_error = 0
        print("  PID controller state has been reset.")
        return True

    def set_clamping_state(self, enabled):
        """Sets the clamping state of the controller for the next run.

        Args:
            enabled (bool): The desired clamping state (True or False).

        Returns:
            bool: True to confirm the method ran successfully.
        """
        self.clamping_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"  Controller output clamping has been {status}.")
        return True

    def update(self, measurement):
        """Calculates the control force based on the current measurement.

        Args:
            measurement (float): The current error measurement from the system.

        Returns:
            float: The final calculated control force (clamped or unclamped).
        """
        error = measurement
        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        output_force = (self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative_error)
        if self.clamping_enabled:
            final_force = np.clip(output_force, -self.max_force, self.max_force)
        else:
            final_force = output_force
        self.previous_error = error
        return final_force

pid_controller = PIDControllerCP(Kp=44, Ki=80, Kd=6, dt=0.001)

def controller_call(measurement):
    """A wrapper function to call the PID controller's update method.

    Args:
        measurement (float): The current error measurement from the system.

    Returns:
        float: The calculated control force from the PID controller.
    """
    force = pid_controller.update(measurement)
    return force

def reset_controller():
    """A wrapper function to allow external calls to the reset method.

    Returns:
        bool: True if the controller was successfully reset.
    """
    return pid_controller.reset()

def set_clamping(enabled):
    """A wrapper function to allow external calls to set the clamping state.

    Args:
        enabled (bool): The desired clamping state (True or False).

    Returns:
        bool: True if the clamping state was successfully set.
    """
    return pid_controller.set_clamping_state(enabled)