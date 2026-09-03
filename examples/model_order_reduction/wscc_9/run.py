"""
This script demonstrates how to create a reduced-order model of the WSCC (Western System Coordinated Council)
9 bus test system and how to simulate dynamics with the resulting reduced-order model.
"""
import os

from sting import datasets, main
from sting.modules.model_order_reduction.balanced_truncation import BalancedTruncation
from sting.modules.model_order_reduction.interconnected_balanced_truncation import (
    InterconnectedBalancedTruncation,
)
from sting.modules.model_order_reduction.singular_perturbation import (
    SingularPerturbation,
)
from sting.utils.dynamical_systems import smooth_step

# Location of all outputs from this example
case_directory = os.path.join(os.getcwd(), "examples", "model_order_reduction", "wscc_9")
os.makedirs(case_directory, exist_ok=True)

# Load the WSCC 9 bus system from the default datasets in STING
system = datasets.wscc_9(case_directory=case_directory)
# Apply any post initialization "updates" of system components
system.apply("post_system_init", system)

# Construct a small-signal model---i.e., full-order model (FOM).
system, ssm = main.run_ssm(system=system)

# Create a reduced order model of all components in the zone labeled as "external".
# We will then connect this reduced order model to the zone labeled "study", which
# consists of a grid forming inverter (GFMI 18A) at bus 2.

zone_name = "external"  # Zone to reduce
r = 12                  # Target reduction order of the external zone

# Vanilla balanced truncation removing the states that are hardest to control and observe.
balanced_truncation = {zone_name:  BalancedTruncation(r=r, method="singular perturbation", library='scipy')}

# Singular perturbation converting the fastest modes to algebraic 
singular_perturbation = {zone_name: SingularPerturbation(r=r, basis="eigen")}

# Interconnected balanced reduction using the *system-level* state-space model to 
# compute a reduced order model. This functions works a little differently as it 
# takes a centralized perspective/approach. We will also use "singular perturbation"
# to eliminate states in order to enforce zero steady-state error---at the expense of 
# accuracy in higher-frequency dynamics. 
interconnected_reduction = InterconnectedBalancedTruncation(r={zone_name:r}, method="singular perturbation")

# Construct a reduced-order model (ROM).
rom1 = main.run_model_reduction(ssm=ssm, reductions=balanced_truncation)
rom2 = main.run_model_reduction(ssm=ssm, reductions=singular_perturbation)
rom3 = interconnected_reduction.reduce(ssm)

# COMPARE the dynamics of a step change to the power reference set points of the 
# grid forming inverter (GFLI 18A) at bus 2
inputs = {
    'gfmi_18a_0': {
        'p_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=0.10, transient_width=5e-3),
        'q_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=-0.10, transient_width=5e-3)
        }
}
t_max = 1.5 # Simulation length in seconds

# Simulate the full-order model
ssm.output_directory = os.path.join(case_directory, "outputs", "full_order_model_simulation")
os.makedirs(ssm.output_directory , exist_ok=True)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

# Simulate the reduced-order models
rom1.output_directory = os.path.join(case_directory, "outputs", "balanced_truncation_simulation")
os.makedirs(rom1.output_directory , exist_ok=True)
rom1.simulate_ssm(t_max=t_max, inputs=inputs)

rom2.output_directory = os.path.join(case_directory, "outputs", "singular_perturbation_simulation")
os.makedirs(rom2.output_directory , exist_ok=True)
rom2.simulate_ssm(t_max=t_max, inputs=inputs)

rom3.output_directory = os.path.join(case_directory, "outputs", "interconnection_reduction_simulation")
os.makedirs(rom3.output_directory , exist_ok=True)
rom3.simulate_ssm(t_max=t_max, inputs=inputs)
