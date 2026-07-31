"""
This script demonstrates how to create a small-signal model of the WSCC (Western System Coordinated Council)
9 bus test system and how to simulate dynamics with the resulting state-space model.
"""
import os

from sting import main, datasets
from sting.utils.dynamical_systems import smooth_step

# Location of all outputs from this example
case_directory = os.path.join(os.getcwd(), "examples", "small_signal_models", "wscc_9")
os.makedirs(case_directory, exist_ok=True)

# Load the WSCC 9 bus system from the default datasets in STING
system = datasets.wscc_9(case_directory=case_directory)
# Apply any post initialization "updates" of system components
system.apply("post_system_init", system)

# Construct a small-signal model
system, ssm = main.run_ssm(system=system, case_directory=case_directory)

# Simulate dynamics of a step change to the power reference set points of the 
# grid forming inverter (GFLI 18A) at bus 2 
inputs = {
    'gfmi_18a_0': {
        'p_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=0.10, transient_width=5e-3),
        'q_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=-0.10, transient_width=5e-3)
        }
}

t_max = 1.5 # Simulation length in seconds
ssm.simulate_ssm(t_max=t_max, inputs=inputs)