"""

Small signal analysis of GFMIE v INF source under variation in Pload, Pref, and Qref. 

""" 
    
# Import Python standard and third-party packages
from pathlib import Path
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Import sting package
from sting.system.core import System
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
from sting.modules.small_signal_modeling.core import SmallSignalModel

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model

inputs = {}
t_max = 2.0

# Load system from CSV files
sys = System.from_csv(case_directory=case_dir)

# Define range of P and Q to vary over 
n = 11
Prange = np.linspace(100,0,n)
Qrange = np.linspace(-50,50,n)

Sbase = 100 # MVA
vdc_ref = 1.05 # pu

def run_sim(sys, P, Q, Sbase, vdc_ref, factor):
    sys.gfmi_e[0].minimum_active_power_MW = -P*factor 
    sys.gfmi_e[0].maximum_active_power_MW = -P*factor 
    sys.gfmi_e[0].minimum_reactive_power_MVAR = Q
    sys.gfmi_e[0].maximum_reactive_power_MVAR = Q
    sys.gfmi_e[0].i_load_ref = P/Sbase/vdc_ref 
    
    # Run power flow
    pf = ACPowerFlow(system=sys, model_settings=None, solver_settings=None)
    pf.solve()
        
    # Break down lines into branches and shunts for small-signal modeling
    sys_modifier = SystemModifier(system=sys)
    sys_modifier.decompose_lines()
    sys_modifier.combine_shunts()

    # Construct small-signal model
    ssm = SmallSignalModel(system=sys)
    ssm.construct_system_ssm()
    return ssm 

# Case 1: Pref = Pload 
results = np.zeros([n,n])
i = 0
for P in Prange:
    j = 0 
    for Q in Qrange: 
        ssm = run_sim(sys, P, Q, Sbase, vdc_ref, 1.0)
        
        max_eig = np.max(np.linalg.eigvals(ssm.model.A).real) 
        
        results[i,j] = max_eig 
        j += 1
    i += 1
    
# Plot results 
fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(30,10))

# Colorbar - use the same range for each
vmin = -6
vmax = 0 
# Plot small-signal stability heatmap 
sns.heatmap(results, xticklabels=Qrange, yticklabels=Prange, linewidth=0.5, ax=axes[1], vmin=vmin, vmax=vmax, cbar=False, cbar_kws = {'label': 'maximum eig. real part'})
axes[1].set_xlabel("Q_sh (MVAR)")
axes[1].set_ylabel("P_load (MW)")
axes[1].set_title("Pref = Pload")

## Case 2: Pref = 0.5*Pload 
results = np.zeros([n,n])
i = 0
for P in Prange:
    j = 0 
    for Q in Qrange: 
        ssm = run_sim(sys, P, Q, Sbase, vdc_ref, 0.5)
        
        max_eig = np.max(np.linalg.eigvals(ssm.model.A).real) 
        
        results[i,j] = max_eig 
        j += 1
    i += 1


sns.heatmap(results, xticklabels=Qrange, yticklabels=Prange, linewidth=0.5, ax=axes[0], vmin=vmin, vmax=vmax, cbar=False, cbar_kws = {'label': 'maximum eig. real part'})
axes[0].set_xlabel("Q_sh (MVAR)")
axes[0].set_ylabel("P_load (MW)")
axes[0].set_title("Pref = 0.5*Pload")

## Case 3: Pref = 1.5*Pload 
results = np.zeros([n,n])
i = 0
for P in Prange:
    j = 0 
    for Q in Qrange: 
        ssm = run_sim(sys, P, Q, Sbase, vdc_ref, 1.5)
        
        max_eig = np.max(np.linalg.eigvals(ssm.model.A).real) 
        
        results[i,j] = max_eig 
        j += 1
    i += 1

sns.heatmap(results, xticklabels=Qrange, yticklabels=Prange, linewidth=0.5,ax=axes[2], vmin=vmin, vmax=vmax, cbar=True, cbar_kws = {'label': 'maximum eig. real part'})
axes[2].set_xlabel("Q_sh (MVAR)")
axes[2].set_ylabel("P_load (MW)")
axes[2].set_title("Pref = 1.5*Pload")
fig.suptitle("Effect of Pref v Pload across P & Q space")

plt.show()
plt.savefig(str(case_dir)+"/ssm_heatmaps.png")

print('ok')
    