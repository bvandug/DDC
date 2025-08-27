# PIDControllerCP.py

import numpy as np

class PIDControllerCP:
    '''
    A class that implements a simple Proportional-Integral-Derivative (PID) controller.
    The output clamping can be enabled or disabled.
    '''
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = 0
        self.previous_error = 0
        self.max_force = 10  # Define the maximum force limit in Newtons
        self.clamping_enabled = True # Flag to control clamping

    def reset(self):
        """Resets the controller's internal state for a new episode."""
        self.integral_error = 0
        self.previous_error = 0
        print("  PID controller state has been reset.")
        return True

    def set_clamping_state(self, enabled):
        """Sets the clamping state of the controller."""
        self.clamping_enabled = enabled
        status = "enabled" if enabled else "disabled"
        print(f"  Controller output clamping has been {status}.")
        return True

    def update(self, measurement):
        error = measurement
        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        
        # Calculate the raw output force
        output_force = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)
                        
        # **--- CONDITIONAL CLAMPING LOGIC ---**
        if self.clamping_enabled:
            # Clamp the output force if enabled
            final_force = np.clip(output_force, -self.max_force, self.max_force)
        else:
            # Use the raw force if disabled
            final_force = output_force
        
        self.previous_error = error
        
        # Return the final force
        return final_force

# Instantiate PIDControllerCP object.
pid_controller = PIDControllerCP(Kp=44, Ki=80, Kd=6, dt=0.001)

def controller_call(measurement):
    """Calls the PID controller to compute the control signal."""
    force = pid_controller.update(measurement)
    return force

def reset_controller():
    """Allows MATLAB to call the reset method."""
    return pid_controller.reset()

def set_clamping(enabled):
    """Allows MATLAB/Python to set the clamping state."""
    return pid_controller.set_clamping_state(enabled)