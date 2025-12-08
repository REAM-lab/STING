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
sys, ssm = main.run_ssm(case_dir)

# Simulate step responses in the system-wide small-signal model
# Actually, we are not exciting the system as u = 0.
# We are testing the "sim" functionality in STING.
tps = np.arange(0, 1.001, 0.001)
def u_func(t):
    return np.zeros((ssm.u.n_device,))

# Simulate small-signal system
dx = ssm.sim(tps, u_func)

# Add initial condition. 
# We take transpose of initial condition vector to fit in the dimension of dx
x_ssm = dx + np.atleast_2d(ssm.x.init).T

# Save results
np.savetxt(os.path.join(case_dir, 'outputs', 'small_signal_model', 'sim_ssm.csv'), 
                        x_ssm.T, delimiter=',', fmt='%f')

# -------------------------------------------------------------
# The rest of this code can be commented. The purpose is to run 
# EMT simulation in Simulink.

# Send system information to MATLAB session (optional)
# First, in MATLAB, execute: matlab.engine.shareEngine('s1')
matlab_session = 's1'
sys.to_matlab(session_name = matlab_session, excluded_attributes=['ssm'])

# Set discretization time in Simulink 
eng = matlab.engine.connect_matlab(matlab_session)
eng.workspace['ts'] = 1E-5
eng.quit()

# In MATLAB, after running EMT simulation,
# execute: writematrix(out.simout.Data, 'sps_output.csv');
# Upload csv file of the simulation
filepath = os.path.join(case_dir, "emt", "sps_output.csv")

# Compare initial conditions.
x_emt = np.loadtxt(filepath, delimiter=',').T
print(np.allclose(x_ssm, x_emt,  rtol=1e-05, atol=1e-04))

print('ok')