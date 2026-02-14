"""
Testcase1 simulates a two infinite sources connected via a transmission line.

First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.
"""

# Import Python standard and third-party packages
import numpy as np
from pathlib import Path

# Import sting package
from sting import main
from sting.system.core import System
from sting.modules.power_flow_d import ACPowerFlow

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

ipopt_solver_settings = {
                "solver_name": "mosek_direct",
                "tee": True,
                "solver_options": {},
            }
model_settings = {
        "generator_type_costs": "linear",
        "load_shedding": False}

# Construct system and small-signal model
sys = System.from_csv(case_directory=case_dir)

acopf = ACPowerFlow(system=sys)
acopf.solve()
acopf.load_solution_to_system()



print('ok')