# PIDControllerPendulum.py

class PIDController:
    """
    A class that implements a simple Proportional-Integral-Derivative (PID) controller.
    This controller assumes the setpoint is 0 and includes optional action clipping.
    """
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        
        # Internal state
        self.integral_error = 0
        self.previous_error = 0
        
        # Clipping parameters
        self.enable_clipping = False
        self.action_limit = float('inf')

    def reset(self):
        """Resets the controller's internal state for a new simulation run."""
        self.integral_error = 0
        self.previous_error = 0
        print("  PID controller state has been reset.")
        return True

    def set_clipping_params(self, enabled, limit):
        """Sets the parameters for action clipping for the next run."""
        self.enable_clipping = enabled
        self.action_limit = abs(limit) if enabled else float('inf')

    def update(self, measurement):
        """Calculates the control output based on the current measurement."""
        # The measurement is treated as the error, since the setpoint is 0
        error = 0 - measurement
        
        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        
        # Calculate the total control output (torque/force)
        output_force = (self.Kp * error +
                        self.Ki * self.integral_error +
                        self.Kd * derivative_error)
        
        self.previous_error = error
        return output_force

# --- TUNE THESE GAINS ---
pid_instance = PIDController(Kp=2, Ki=0, Kd=0.3, dt=0.001)

def controller_call(measurement):
    """
    Function called by Simulink to get the control signal.
    This function now handles the clipping logic.
    """
    raw_force = pid_instance.update(measurement)
    
    if pid_instance.enable_clipping:
        clipped_force = max(min(raw_force, pid_instance.action_limit), -pid_instance.action_limit)
        return clipped_force
    else:
        return raw_force

def reset_controller():
    """Function called by Simulink to reset the controller state between runs."""
    return pid_instance.reset()

def configure_controller_clipping(enabled, limit):
    """
    Function called by the main Python script to configure clipping
    before a simulation run starts.
    """
    pid_instance.set_clipping_params(enabled, limit)
    return True