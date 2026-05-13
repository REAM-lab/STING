"""
Simulates an infinite source and GFLI_d connected via a transmission line.

"""
# Import Python standard and third-party packages
from pathlib import Path
import sys 
sys.path.append("/Users/ruthkravis/Documents/STING")

import numpy as np
import pandas as pd 
from scipy.linalg import eig, inv
import seaborn as sns 
import matplotlib.pyplot as plt

# Import sting package
from sting import main
from sting.system.core import System

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
sys = System.from_csv(case_directory=case_dir)

# Step function to simulate
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

inputs = {
    'infinite_sources_0': {
        'v_ref_d': step2
        }, 
    'gfli_d_0': {
        'i_bus_d_ref': step1}
    }
t_max = 2.0 # Simulation length

# Construct system and small-signal model
_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation (not implemented yet for gfli_a)
main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

# def participation_factor_plot(ssm):
#     w,vr = eig(ssm.model.A)
#     vl = inv(vr) # ensures normalization 

#     p = np.zeros_like(vl)
#     for i in range(len(w)):
#         for k in range(len(w)):
#             # use correct formula for complex eigenvalues 
#             p[k, i] = (vl[i,k].real)**2/(np.dot(vl[i,:].real,vl[i,:].real)) # participation of k-th state (row) in i-th mode (column)
            
#     pf_df = pd.DataFrame(p.real, index=ssm.model.x.name)
#     fig = plt.figure(figsize=(12,8))
#     ax = sns.heatmap(pf_df, yticklabels=pf_df.index, xticklabels=np.round(w.real,3), linewidths=1, linecolor='white')
#     ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
#     plt.show()
#     plt.savefig("ssm.png")
    
    
# participation_factor_plot(ssm)
    

print('\nok')