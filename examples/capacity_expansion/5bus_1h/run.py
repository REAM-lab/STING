# Import Python standard and third-party packages
import os
import matlab.engine
import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np

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
        "generator_type_costs": "quadratic",
        "load_shedding": False,
        "line_capacity_expansion": False,
        "line_capacity": False,
        "kron_equivalent_flow_constraints": False,
        "bus_max_flow_expansion": False,
        "bus_max_flow": True,
        "policies": [],
    }

gurobi_solver_settings = {
                "solver_name": "gurobi",
                "tee": True,
                "solver_options": {'BarHomogeneous':1,
                                   'FeasibilityTol':1e-5,
                                   'CrossOver':0,
			           'Method': 2},
}

main.run_capex(case_dir, solver_settings=mosek_solver_settings, model_settings=model_settings)

main.run_zonal_capex(case_dir, solver_settings=mosek_solver_settings, model_settings=model_settings, components_to_clone=['cf', 'tp', 'ts', 'sc'])

print('ok')