"""
Testcase4 simulates a GFMIc connected to an infinite source
via a transmission line.

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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# Import sting package
from sting import main

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent
print(case_dir)
# Construct system and small-signal model
sys, ssm = main.run_ssm(case_dir)

# Simulate step responses in the system-wide small-signal model
tps = np.arange(0, 3.001, 0.001)
def u_func(t):
    p_ref = 0.1 if t>= 0.3 else 0
    q_ref = -0.1 if t>= 2 else 0
    v_ref = 0 
    v_int_d = 0
    v_int_q = 0
    return np.array([p_ref, q_ref, v_ref, v_int_d, v_int_q])

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

# Send system information to a MATLAB session (optional)
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
x_emt = np.loadtxt(filepath, delimiter=',').T

# Compare x_emt and x_ssm
states = ssm.x.name
dir = os.path.join(case_dir, "outputs", "small_signal_model")

for (i, x) in enumerate(states):
    fig, ax = plt.subplots(dpi=1200, figsize=(4,3.0))
    ax.plot(tps, x_ssm[i], color='red', linestyle='solid')
    ax.plot(tps, x_emt[i], color='blue', linestyle='dashed')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(f'{x}')
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{x}.pdf"))

print('ok')
