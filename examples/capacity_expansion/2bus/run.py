"""
This script runs the capacity expansion model using transport model for a 2-bus system.
Objective function value: 2000.

Author: Paul Serna-Torre
Date: 2026-02-15

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
                "solver_options": {},
            }
model_settings = {
        "generation_capacity_expansion": False,
        "storage_capacity_expansion": False,
        "line_capacity_expansion": False,
        "line_capacity": True,
        "power_flow": "transport",
    }

gurobi_solver_settings = {
                "solver_name": "gurobi",
                "tee": True,
                "solver_options": {'BarHomogeneous':1,
                                   'FeasibilityTol':1e-5,
                                   'CrossOver':0,
			           'Method': 2},
}

capex, system = main.run_capex(case_dir, solver_settings=mosek_solver_settings, model_settings=model_settings)

print('ok')