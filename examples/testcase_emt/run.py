"""
Testcase1 simulates a two infinite sources connected via a transmission line.

First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.

Second (optional), we transfer the system data and initial conditions 
to Simulink. STING has a functionality to transfer this data. 
In EMT simulation is in the file sim_emt.slx. 

Comparison between the small-signal simulation and EMT simultion
shows a proximity of these domain responses. Dec 7, 2025.
"""

# Import Python standard and third-party packages
import os
import matlab.engine
import numpy as np
import pandas as pd

# Import sting package
from sting import main

# Specify path of the case study directory
case_dir = os.path.join(os.getcwd(), "examples", "testcase1")

# Construct system and small-signal model
def step1(t):
    return 0.0

inputs = {'inf_src_1': {'v_ref_d': step1}, 
          'inf_src_2': {'v_ref_d': step1}}

t_max = 1.0

solution = main.run_emt(t_max, inputs, case_dir =case_dir)

# Define timepoints that will be used to evaluate the solution of the ODEs
tps = np.linspace(0, t_max, 500)
n_tps = len(tps)

# Extract solution of the ODEs and evaluate at the timepoints
interp_sol = solution.sol(tps)
angle_pc = interp_sol[0]

print('ok')