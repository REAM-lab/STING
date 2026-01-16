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
                "gen_costs": "quadratic",
                "consider_shedding": True,
                "consider_line_capacity": False,
                "consider_bus_max_flow": True,
                "consider_angle_limits": True,
            }

gurobi_solver_settings = {
                "solver_name": "gurobi",
                "tee": True,
                "solver_options": {'BarHomogeneous':1,
                                   'FeasibilityTol':1e-5,
                                   'CrossOver':0,
			           'Method': 2},
}

sys= main.run_capex(case_dir, solver_settings=mosek_solver_settings, model_settings=model_settings)

print('ok')