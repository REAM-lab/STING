"""
This script runs a capacity expansion model for the 5-bus case study with 1-hour time resolution.
It enforces a policy that limits the total built transmission capacity to 15 MW and 10 MW, respectively.

total_cost_USD:
tx_cap_10: 4177803.4478469538 USD
tx_cap_15: 4177498.5217958856 USD
"""

# Import Python standard and third-party packages
from pathlib import Path
import os

# Import sting package
from sting import main

# Import policy component
from sting.policies.transmission_expansion_constraint.core import TransmissionExpansionConstraint

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

mosek_solver_settings = {
        "solver_name": "mosek_direct",
        "tee": True,
        "solver_options": {
            'MSK_DPAR_INTPNT_TOL_PFEAS':'1e-5',
        },
    }

model_settings = {
        "load_shedding": True,
        "line_capacity_expansion": True,
        "line_capacity": True,
        "power_flow": "dc",
    }

sys = main.run_capex(
    case_directory=case_dir, 
    solver_settings=mosek_solver_settings, 
    model_settings=model_settings,
    components_to_add=[TransmissionExpansionConstraint(built_transmission_capacity_cap_MW=15)],
    output_directory=os.path.join(case_dir, "outputs", "tx_cap_15"))

# If you want to keep running the model with a different transmission capacity, just add like this:
'''
sys = main.run_capex(
    case_directory=case_dir, 
    solver_settings=mosek_solver_settings, 
    model_settings=model_settings,
    components_to_add=[TransmissionExpansionConstraint(built_transmission_capacity_cap_MW=10)],
    output_directory=os.path.join(case_dir, "outputs", "tx_cap_10"))
'''
print('ok')
