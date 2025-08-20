class PIDControllerPend:
    '''
    A class that implements a simple Proportional-Integral-Derivative (PID) controller.
    '''
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = 0
        self.previous_error = 0

    # --- ADD THIS NEW METHOD ---
    def reset(self):
        """Resets the controller's internal state for a new episode."""
        self.integral_error = 0
        self.previous_error = 0
        print("  PID controller state has been reset.")
        return True

    def update(self, measurement):
        error = measurement
        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        output_force = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)
        self.previous_error = error
        return output_force

# Instantiate PIDControllerPend object.
pid_controller = PIDControllerPend(Kp=44, Ki=80, Kd=6, dt=0.001)

def controller_call(measurement):
    """Calls the PID controller to compute the control signal."""
    force = pid_controller.update(measurement)
    return force

# --- ADD THIS NEW FUNCTION ---
def reset_controller():
    """Allows MATLAB to call the reset method."""
    return pid_controller.reset()