# Import Python standard and third-party packages
import os
import matlab.engine
import numpy as np
import pandas as pd
from pathlib import Path

# Import sting package
from sting import main

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
mosek_solver_settings = {
                "solver_name": "mosek_direct",
                "tee": True,
                "solver_options": {'INTPNT_TOL_PFEAS':1e-5},
            }
model_settings = {
                "gen_costs": "linear",
                "consider_shedding": False,
            }

gurobi_solver_settings = {
                "solver_name": "gurobi",
                "tee": True,
                "solver_options": {'BarHomogeneous':1,
                                   'FeasibilityTol':1e-5,
                                   'Method':2,
                                   'CrossOver':0},
}

sys= main.run_capex(case_dir, model_settings=model_settings, solver_settings=gurobi_solver_settings)

print('ok')