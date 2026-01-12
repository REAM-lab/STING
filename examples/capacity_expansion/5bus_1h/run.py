# Import Python standard and third-party packages
import os
import matlab.engine
import numpy as np
import pandas as pd
from pathlib import Path


import numpy as np

# Sample NumPy array
a = np.array([
    [1, 0, 0+0j, 0, 0], # 2 non-zero entries
    [2, 3, 4, 0, 1], # 4 non-zero entries (should be selected)
    [0, 0, 0, 0, 0], # 0 non-zero entries
    [0, 1, 1, 0, 1]  # 3 non-zero entries (should be selected)
])

# Count non-zero entries per row
# (a != 0) creates a boolean array, and .sum(axis=1) counts True values
non_zero_counts = (~np.isclose(a, 0)).sum(axis=1)
mask = non_zero_counts > 2
# Alternative using np.count_nonzero

# Import sting package
from sting import main

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

mosek_solver_settings = {
                "solver_name": "mosek_direct",
                "tee": True,
                "solver_options": {'MSK_DPAR_INTPNT_TOL_PFEAS':'1e-5'},
            }
model_settings = {
                "gen_costs": "quadratic",
                "consider_shedding": True,
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