"""

Analysis of EMT response of GFMI_E v INF bus system to a grid disturbance under different P/Q and different Pref/Pload relationships

In progress - April 3 (Ruth)

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
from sting.modules.simulation_emt.core import SimulationEMT
from sting.modules.small_signal_modeling.core import SmallSignalModel

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
def step1(t):
    return 0.3 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

def step3(t):
    return 0.1 if t > 1.0 else 0.0 

inputs = {'infinite_sources_0': {'v_ref_d': step1}, 
          'gfmi_e_0': {'p_ref': step2, 
                       'q_ref': step2,
                       'v_ref': step2,
                       'v_dc_ref': step2,
                       'v_s': step2, 
                       'i_load_ref': step2}}

t_max = 2.0

# Load system from CSV files
sys = System.from_csv(case_directory=case_dir)

n = 3
Prange = np.linspace(100,0,n)
Qrange = np.linspace(-50,50,n)

Sbase = 100 # MVA
vdc_ref = 1.05 # pu 

tps = np.linspace(0, t_max, 500)
dt = tps[1] - tps[0]

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
    
    emt_sc = SimulationEMT(system=sys)
    emt_sc.sim(t_max, inputs)
    return emt_sc        
        
        
# Case 1: Pref = Pload 
results_rocof = np.zeros([Prange.shape[0],Qrange.shape[0]])
results_nadir = np.zeros([Prange.shape[0],Qrange.shape[0]])

i = 0
for P in Prange:
    j = 0 
    for Q in Qrange: 
        emt_sc = run_sim(sys, P, Q, Sbase, vdc_ref, 1.0)
        
        # Measure some performance metrics - frequency ROCOF and nadir and steady state error ? 
        gfm = emt_sc.system.gfmi_e[0].variables_emt.x.value
        w_pc = gfm[1,:]
        # nadir 
        nadir = np.min(w_pc)
        results_nadir[i,j] = nadir 
        # rocof - calculate as a moving average of duration 100ms (~25 timesteps for us)
        df_dt = np.abs(np.diff(w_pc)/dt) # calculate raw rocof 
        df_dt_ma = np.convolve(df_dt,np.ones(25)/25) # get moving averages over 25 time steps (~100ms)
        rocof = np.max(df_dt_ma)
        
        results_rocof[i,j] = rocof  
        j += 1
    i += 1
    
# Plot results 
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(30,15))

sns.heatmap(results_nadir, xticklabels=Qrange, yticklabels=Prange, linewidth=0.5, ax=axes[0,1], cbar=True, cbar_kws = {'label': 'nadir'})
axes[0,1].set_xlabel("Q_sh")
axes[0,1].set_ylabel("P_load")
axes[0,1].set_title("Pref = Pload")

sns.heatmap(results_rocof, xticklabels=Qrange, yticklabels=Prange, linewidth=0.5, ax=axes[1,1], cbar=True, cbar_kws = {'label': 'rocof'})
axes[1,1].set_xlabel("Q_sh")
axes[1,1].set_ylabel("P_load")
axes[1,1].set_title("Pref = Pload")


# Case 2: Pref < Pload (Pref=0.8*Pload)
    
results_rocof = np.zeros([Prange.shape[0],Qrange.shape[0]])
results_nadir = np.zeros([Prange.shape[0],Qrange.shape[0]])

i = 0
for P in Prange:
    j = 0 
    for Q in Qrange: 
        emt_sc = run_sim(sys, P, Q, Sbase, vdc_ref, 0.8)
        
        # Measure some performance metrics - frequency ROCOF and nadir and steady state error ? 
        gfm = emt_sc.system.gfmi_e[0].variables_emt.x.value
        w_pc = gfm[1,:]
        # nadir 
        nadir = np.min(w_pc)
        results_nadir[i,j] = nadir 
        # rocof - calculate as a moving average of duration 100ms (~25 timesteps for us)
        df_dt = np.abs(np.diff(w_pc)/dt) # calculate raw rocof 
        df_dt_ma = np.convolve(df_dt,np.ones(25)/25) # get moving averages over 25 time steps (~100ms)
        rocof = np.max(df_dt_ma)
        
        results_rocof[i,j] = rocof  
        j += 1
    i += 1
    
# Plot results 

sns.heatmap(results_nadir, xticklabels=Qrange, yticklabels=Prange, linewidth=0.5, ax=axes[0,0], cbar=True, cbar_kws = {'label': 'nadir'})
axes[0,0].set_xlabel("Q_sh")
axes[0,0].set_ylabel("P_load")
axes[0,0].set_title("Pref = 0.8*Pload")

sns.heatmap(results_rocof, xticklabels=Qrange, yticklabels=Prange, linewidth=0.5, ax=axes[1,0], cbar=True, cbar_kws = {'label': 'rocof'})
axes[1,0].set_xlabel("Q_sh")
axes[1,0].set_ylabel("P_load")
axes[1,0].set_title("Pref = 0.8*Pload")


# Case 3: Pref > Pload (Pref=1.2*Pload)

results_rocof = np.zeros([Prange.shape[0],Qrange.shape[0]])
results_nadir = np.zeros([Prange.shape[0],Qrange.shape[0]])

i = 0
for P in Prange:
    j = 0 
    for Q in Qrange: 
        emt_sc = run_sim(sys, P, Q, Sbase, vdc_ref, 1.2)
        
        # Measure some performance metrics - frequency ROCOF and nadir and steady state error ? 
        gfm = emt_sc.system.gfmi_e[0].variables_emt.x.value
        w_pc = gfm[1,:]
        # nadir 
        nadir = np.min(w_pc)
        results_nadir[i,j] = nadir 
        # rocof - calculate as a moving average of duration 100ms (~25 timesteps for us)
        df_dt = np.abs(np.diff(w_pc)/dt) # calculate raw rocof 
        df_dt_ma = np.convolve(df_dt,np.ones(25)/25) # get moving averages over 25 time steps (~100ms)
        rocof = np.max(df_dt_ma)
        
        results_rocof[i,j] = rocof  
        j += 1
    i += 1
    
# Plot results 

sns.heatmap(results_nadir, xticklabels=Qrange, yticklabels=Prange, linewidth=0.5, ax=axes[0,2], cbar=True, cbar_kws = {'label': 'nadir'})
axes[0,2].set_xlabel("Q_sh")
axes[0,2].set_ylabel("P_load")
axes[0,2].set_title("Pref = 1.2*Pload")

sns.heatmap(results_rocof, xticklabels=Qrange, yticklabels=Prange, linewidth=0.5, ax=axes[1,2], cbar=True, cbar_kws = {'label': 'rocof'})
axes[1,2].set_xlabel("Q_sh")
axes[1,2].set_ylabel("P_load")
axes[1,2].set_title("Pref = 1.2*Pload")

plt.show()
plt.savefig(str(case_dir)+"/emt_heatmaps.png")

print('ok')