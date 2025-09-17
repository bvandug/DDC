# PIDMainBC.py

import matlab.engine
import numpy as np
import matplotlib.pyplot as plt

def setNoise(eng, model, noise_std_dev):
    """Sets the observational noise level in the Simulink model.

    Args:
        eng (matlab.engine.MatlabEngine): The active MATLAB engine instance.
        model (str): The name of the Simulink model.
        noise_std_dev (float): The standard deviation of the desired noise.

    Returns:
        None
    """
    if noise_std_dev > 0:
        noise_variance = noise_std_dev ** 2
        eng.set_param(f'{model}/Random Number', 'Variance', str(noise_variance), nargout=0)
        eng.set_param(f'{model}/Random Number', 'Seed', str(np.random.randint(1, 100000)), nargout=0)
        print(f"  Noise enabled with standard deviation {noise_std_dev}")
    else:
        eng.set_param(f'{model}/Random Number', 'Variance', str(0), nargout=0)
        print("  Noise disabled.")

def calculate_metrics(time_lst, voltage_lst, desired_voltage, stability_threshold=0.5):
    """Calculates performance metrics from the simulation data.

    Args:
        time_lst (list): List of timestamps.
        voltage_lst (list): List of voltage measurements.
        desired_voltage (float): The target voltage for the simulation.
        stability_threshold (float): The error margin to define stability.

    Returns:
        dict: A dictionary containing the calculated performance metrics.
    """
    metrics = {'stabilisation_time_ms': 'Did not stabilize', 'overshoot_v': 0.0, 'steady_state_error_v': 'N/A'}
    max_voltage = np.max(voltage_lst)
    if max_voltage > desired_voltage:
        metrics['overshoot_v'] = max_voltage - desired_voltage
    stable_band_min = desired_voltage - stability_threshold
    stable_band_max = desired_voltage + stability_threshold
    time_delta = time_lst[1] - time_lst[0]
    required_stable_steps = int(0.01 / time_delta)
    consecutive_stable_steps = 0
    first_stable_time = None
    for i in range(len(voltage_lst)):
        voltage = voltage_lst[i]
        if stable_band_min <= voltage <= stable_band_max:
            if consecutive_stable_steps == 0:
                first_stable_time = time_lst[i]
            consecutive_stable_steps += 1
            if consecutive_stable_steps >= required_stable_steps:
                metrics['stabilisation_time_ms'] = first_stable_time * 1000
                settled_region = voltage_lst[int(len(voltage_lst) * 0.8):]
                metrics['steady_state_error_v'] = np.mean(np.abs(np.array(settled_region) - desired_voltage))
                break
        else:
            consecutive_stable_steps = 0
            first_stable_time = None
    return metrics

def main(desired_voltage=30.0, source_voltage=48.0, model='bcSimPID', stabilisation_precision=0.5):
    """Sets up MATLAB, runs simulations, and calculates performance metrics.

    Args:
        desired_voltage (float): The target output voltage.
        source_voltage (float): The input voltage to the converter.
        model (str): The name of the Simulink model file.
        stabilisation_precision (float): Error margin for stability checks.

    Returns:
        None
    """
    noise_levels_to_test = [0.0, 0.001, 0.01, 0.1]
    num_episodes = 1
    print("Setting up engine...")
    eng = matlab.engine.start_matlab()
    eng.load_system(model, nargout=0)
    eng.set_param(f'{model}/finalVoltage', 'Value', str(desired_voltage), nargout=0)
    eng.set_param(f'{model}/input_voltage', 'Amplitude', str(source_voltage), nargout=0)
    for noise_level in noise_levels_to_test:
        print(f"\n--- Testing Noise Level σ = {noise_level} ---")
        all_episode_metrics = []
        for episode in range(1, num_episodes + 1):
            print(f"  Running episode {episode}/{num_episodes}...")
            eng.py.PIDControllerBC.reset_controller(nargout=0)
            setNoise(eng, model, noise_level)
            eng.eval(f"out = sim('{model}');", nargout=0)
            voltage_2d = eng.eval("voltage")
            voltage_lst = [v[0] for v in voltage_2d]
            time_2d = eng.eval("time")
            time_lst = [t[0] for t in time_2d]
            metrics = calculate_metrics(time_lst, voltage_lst, desired_voltage)
            all_episode_metrics.append(metrics)
            print(f"    - Stabilisation Time: {metrics['stabilisation_time_ms']:.2f} ms" if isinstance(metrics['stabilisation_time_ms'], float) else f"    - Stabilisation Time: {metrics['stabilisation_time_ms']}")
            print(f"    - Overshoot: {metrics['overshoot_v']:.2f} V")
            print(f"    - Steady-State Error: {metrics['steady_state_error_v']:.4f} V" if isinstance(metrics['steady_state_error_v'], float) else f"    - Steady-State Error: {metrics['steady_state_error_v']}")
        valid_stab_times = [m['stabilisation_time_ms'] for m in all_episode_metrics if isinstance(m['stabilisation_time_ms'], float)]
        avg_stab_time = np.mean(valid_stab_times) if valid_stab_times else 'N/A'
        std_stab_time = np.std(valid_stab_times) if valid_stab_times else 'N/A'
        avg_overshoot = np.mean([m['overshoot_v'] for m in all_episode_metrics])
        std_overshoot = np.std([m['overshoot_v'] for m in all_episode_metrics])
        valid_sse = [m['steady_state_error_v'] for m in all_episode_metrics if isinstance(m['steady_state_error_v'], float)]
        avg_sse = np.mean(valid_sse) if valid_sse else 'N/A'
        std_sse = np.std(valid_sse) if valid_sse else 'N/A'
        print("\n  Average Performance Metrics:")
        print(f"    - Avg. Stabilisation Time: {avg_stab_time:.2f} ± {std_stab_time:.2f} ms" if isinstance(avg_stab_time, float) else f"    - Avg. Stabilisation Time: {avg_stab_time}")
        print(f"    - Avg. Overshoot: {avg_overshoot:.2f} ± {std_overshoot:.2f} V")
        print(f"    - Avg. Steady-State Error: {avg_sse:.4f} ± {std_sse:.4f} V" if isinstance(avg_sse, float) else f"    - Avg. Steady-State Error: {avg_sse}")
    eng.quit()

if __name__ == '__main__':
    main()