"""
This script runs an operations model for a 5-bus system enforcing energy
and power budget constraints on a set of hydro generators. Both power
and energy budgets are defined using terms--a set of timepoints in which
the constraints are active. Here we enforce a 600MW power constraint (i.e.,
all hydro generators cannot dispatch more than 600MW in total at any hour) 
and a 1.7GWh energy constraint (i.e., over the 3 timepoints hydro energy cannot 
dispatch more than 1.7GW). Both constraints are binding in this example.

Author: Adam Sedlak
Date: 2026-02-13
"""

# Import Python standard and third-party packages
from pathlib import Path

# Import sting package
from sting import main

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
        "line_capacity_expansion": False,
        "line_capacity": True,
        "power_flow": "dc",
        "policies": [
            "sting.policies.energy_budgets"]
    }

sys = main.run_capex(
    case_directory=case_dir, 
    solver_settings=mosek_solver_settings, 
    model_settings=model_settings)

print('ok')
