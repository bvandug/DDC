"""
MATLAB vs JAX Precision Test and Configuration Script
Simple version for same-directory setup
"""

import numpy as np
import jax
import jax.numpy as jnp

from ip_simulink_env import SimulinkEnv
from ip_jax_wrapper import InvertedPendulumGymWrapper as JaxEnvWrapper
from ip_jax import PendulumState as JaxPendulumState

# Enforce 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

def test_basic_precision():
    """Test basic floating point operations"""
    print("=== Basic Precision Test ===")
    
    # Test values
    test_cases = [
        (0.1, "Simple decimal"),
        (np.pi/4, "Pi/4"),
        (np.sin(0.5), "sin(0.5)"),
        (np.cos(0.3), "cos(0.3)"),
        (0.2 * 0.15**2, "Moment of inertia calculation")
    ]
    
    print(f"{'Description':<25} {'Python Value':<20} {'Precision'}")
    print("-" * 65)
    
    for val, desc in test_cases:
        print(f"{desc:<25} {val:<20.15f} {val:.2e}")

def configure_matlab_precision(eng, model_name):
    """Configure MATLAB for maximum precision"""
    print("\n=== Configuring MATLAB Precision ===")
    
    try:
        # Set display format for maximum precision
        eng.eval("format long", nargout=0)
        print("✓ Set MATLAB format to 'long'")
        
        # Force double precision data types
        eng.set_param(model_name, 'DefaultDataTypeOverride', 'double', nargout=0)
        eng.set_param(model_name, 'DataTypeOverride', 'UseLocalSettings', nargout=0)
        eng.set_param(model_name, 'DefaultUnderspecifiedDataType', 'double', nargout=0)
        print("✓ Set Simulink data types to double precision")
        
        # Configure solver for maximum precision
        eng.set_param(model_name, 'Solver', 'ode1', nargout=0)
        eng.set_param(model_name, 'FixedStep', '0.01', nargout=0)
        eng.set_param(model_name, 'SolverType', 'Fixed-step', nargout=0)
        eng.set_param(model_name, 'RelTol', '1e-15', nargout=0)
        eng.set_param(model_name, 'AbsTol', '1e-15', nargout=0)
        print("✓ Configured ode1 solver with tight tolerances")
        
        # Set physical constants with high precision
        eng.eval("""
        clear all;
        m = 0.2;
        L = 0.15; 
        g = 9.8;
        dt = 0.01;
        max_torque = 2.0;
        angle_threshold = pi/2;
        """, nargout=0)
        print("✓ Set physical constants in MATLAB workspace")
        
        return True
        
    except Exception as e:
        print(f"❌ Error configuring MATLAB: {e}")
        return False

def test_matlab_precision(eng):
    """Test MATLAB's precision capabilities"""
    print("\n=== MATLAB Precision Test ===")
    
    try:
        # Check machine epsilon
        eps_val = eng.eval("eps", nargout=1)
        print(f"MATLAB machine epsilon: {eps_val:.2e}")
        
        # Test calculations that match our physics
        test_results = {}
        
        # Basic math operations
        test_results['sin_test'] = eng.eval("sin(0.5)", nargout=1)
        test_results['cos_test'] = eng.eval("cos(0.3)", nargout=1)
        test_results['moment_inertia'] = eng.eval("0.2 * 0.15^2", nargout=1)
        test_results['gravity_term'] = eng.eval("-0.2 * 9.8 * 0.15 * sin(0.1)", nargout=1)
        
        # Physics calculation (pendulum acceleration)
        test_results['theta_ddot'] = eng.eval("(-0.2 * 9.8 * 0.15 * sin(0.1) + 1.0) / (0.2 * 0.15^2)", nargout=1)
        
        print(f"{'Test':<20} {'MATLAB Result':<20} {'Python Result':<20} {'Difference'}")
        print("-" * 80)
        
        # Compare with Python calculations
        python_results = {
            'sin_test': np.sin(0.5),
            'cos_test': np.cos(0.3),
            'moment_inertia': 0.2 * 0.15**2,
            'gravity_term': -0.2 * 9.8 * 0.15 * np.sin(0.1),
            'theta_ddot': (-0.2 * 9.8 * 0.15 * np.sin(0.1) + 1.0) / (0.2 * 0.15**2)
        }
        
        max_diff = 0
        for test_name in test_results:
            matlab_val = test_results[test_name]
            python_val = python_results[test_name]
            diff = abs(matlab_val - python_val)
            max_diff = max(max_diff, diff)
            
            print(f"{test_name:<20} {matlab_val:<20.15f} {python_val:<20.15f} {diff:.2e}")
        
        print(f"\nMaximum difference: {max_diff:.2e}")
        
        if max_diff < 1e-14:
            print("✅ Excellent precision match!")
        elif max_diff < 1e-10:
            print("✅ Good precision match")
        else:
            print("⚠️  Precision differences detected")
            
        return test_results, python_results
        
    except Exception as e:
        print(f"❌ Error testing MATLAB precision: {e}")
        return None, None

def run_mini_simulation_comparison():
    """Run a mini simulation to compare integration precision"""
    print("\n=== Mini Simulation Comparison ===")
    
    # Simple pendulum step using identical parameters
    dt = 0.01
    m, L, g = 0.2, 0.15, 9.8
    I = m * L**2
    
    # Initial conditions
    theta = 0.1
    theta_dot = 0.0
    action = 1.0
    
    print(f"Initial: θ={theta:.15f}, θ̇={theta_dot:.15f}")
    print(f"Action: {action:.15f}")
    
    # Manual calculation (what JAX does)
    theta_ddot = (-m * g * L * np.sin(theta) + action) / I
    theta_dot_new = theta_dot + theta_ddot * dt
    theta_new = theta + theta_dot_new * dt
    # Normalize angle
    theta_new_norm = ((theta_new + np.pi) % (2 * np.pi)) - np.pi
    
    print(f"\nPython calculation:")
    print(f"  θ̈ = {theta_ddot:.15f}")
    print(f"  θ̇_new = {theta_dot_new:.15f}")
    print(f"  θ_new = {theta_new:.15f}")
    print(f"  θ_norm = {theta_new_norm:.15f}")
    
    return {
        'theta_ddot': theta_ddot,
        'theta_dot_new': theta_dot_new,
        'theta_new': theta_new,
        'theta_norm': theta_new_norm
    }

def test_simulink_vs_jax():
    """Full test comparing Simulink and JAX environments"""
    print("\n=== Full Environment Comparison ===")
    
    try:
        # Initialize environments
        print("Initializing environments...")
        simulink_env = SimulinkEnv(model_name="pendulum", seed=42)
        jax_env = JaxEnvWrapper(seed=42)
        
        # Configure MATLAB precision
        if configure_matlab_precision(simulink_env.eng, simulink_env.model_name):
            print("✅ MATLAB configured for maximum precision")
        
        # Test MATLAB precision
        matlab_results, python_results = test_matlab_precision(simulink_env.eng)
        
        # Reset environments
        simulink_obs = simulink_env.reset()
        jax_obs, _ = jax_env.reset()
        
        print(f"\nInitial states:")
        print(f"Simulink: θ={simulink_obs[0]:.15f}, θ̇={simulink_obs[1]:.15f}")
        print(f"JAX:      θ={jax_obs[0]:.15f}, θ̇={jax_obs[1]:.15f}")
        
        # Single step comparison
        action = np.array([1.0])
        simulink_obs2, _, _, _ = simulink_env.step(action)
        jax_obs2, _, _, _, _ = jax_env.step(action)
        
        print(f"\nAfter one step:")
        print(f"Simulink: θ={simulink_obs2[0]:.15f}, θ̇={simulink_obs2[1]:.15f}")
        print(f"JAX:      θ={jax_obs2[0]:.15f}, θ̇={jax_obs2[1]:.15f}")
        
        # Calculate differences
        theta_diff = abs(simulink_obs2[0] - jax_obs2[0])
        theta_dot_diff = abs(simulink_obs2[1] - jax_obs2[1])
        
        print(f"\nDifferences after one step:")
        print(f"Δθ = {theta_diff:.2e}")
        print(f"Δθ̇ = {theta_dot_diff:.2e}")
        
        if theta_diff < 1e-12 and theta_dot_diff < 1e-12:
            print("✅ Excellent precision match!")
        elif theta_diff < 1e-8 and theta_dot_diff < 1e-8:
            print("✅ Good precision match")
        else:
            print("⚠️  Precision differences remain")
        
        # Multi-step comparison
        print("\n=== 10-Step Comparison ===")
        
        # Reset again for multi-step test
        simulink_obs = simulink_env.reset()
        jax_env.state = JaxPendulumState(
            theta=simulink_obs[0], 
            theta_dot=simulink_obs[1], 
            t=0.0, 
            done=False
        )
        
        print("Step | Simulink θ        | JAX θ             | Difference")
        print("-" * 60)
        print(f"  0  | {simulink_obs[0]:15.10f} | {simulink_obs[0]:15.10f} | {0:.2e}")
        
        for step in range(10):
            action = np.array([np.sin(step * 0.1) * 2.0])
            simulink_obs, _, _, _ = simulink_env.step(action)
            jax_obs, _, _, _, _ = jax_env.step(action)
            
            diff = abs(simulink_obs[0] - jax_obs[0])
            print(f"  {step+1}  | {simulink_obs[0]:15.10f} | {jax_obs[0]:15.10f} | {diff:.2e}")
        
        # Cleanup
        simulink_env.close()
        jax_env.close()
        
    except Exception as e:
        print(f"❌ Error in environment comparison: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run all precision tests"""
    print("MATLAB vs JAX Precision Diagnostic Tool")
    print("=" * 50)
    
    # Basic precision test
    test_basic_precision()
    
    # Mini simulation comparison
    python_results = run_mini_simulation_comparison()
    
    # Full environment test
    test_simulink_vs_jax()
    
    print("\n" + "=" * 50)
    print("Precision testing complete!")
    print("\nRecommendations:")
    print("1. Run your RL training with the configured MATLAB settings")
    print("2. The remaining small differences should not affect RL performance")
    print("3. Monitor training curves - they should now be much more similar")

if __name__ == "__main__":
    main()