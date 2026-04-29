

# In progress - March 5, 2026 - Ruth 

# Import Python standard and third-party packages
from pathlib import Path
from scipy import signal 
import numpy as np 

# Import sting package
from sting import main
# from sting.system.core import System
# from sting.system.operations import SystemModifier
# from sting.modules.power_flow.core import ACPowerFlow
# from sting.modules.simulation_emt.core import SimulationEMT
# from sting.modules.small_signal_modeling.core import SmallSignalModel

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
def step1(t):
    return 0.5 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

def step3(t):
    return 0.25 if t > 1.0 else 0.0 

def step3_neg(t):
    return -0.25 if t > 1.0 else 0.0 

def square_oscillation(t):
    osc = 0.1*signal.square(2 * np.pi * 14 * t)
    return osc

# Specify inputs to excite - any constant input does not need to be specified 
# NB: input is a perturbation from the nominal value 
inputs = {'infinite_sources_0': {'v_ref_d': step2}, 
          'gfmi_e_0': {'p_ref': step3_neg, 
                       'q_ref': step2,
                       'v_ref': step2,
                       'v_dc_ref': step2,
                       'v_s': step2, 
                       'i_load_ref': step3}}

t_max = 4.0

# Construct system and small-signal model
sys, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation
main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)


# # Load system from CSV files
# sys = System.from_csv(case_directory=case_dir)

# # Match power reference with load 
# P_load = -100 # MW 
# sys.gfmi_e[0].minimum_active_power_MW = -P_load
# sys.gfmi_e[0].maximum_active_power_MW = -P_load
# sys.gfmi_e[0].minimum_reactive_power_MVAR = 0
# sys.gfmi_e[0].maximum_reactive_power_MVAR = 0

# # i_load_ref*v_dc_ref = i_load_ref*1 = p_load = i_load_ref  
# Sbase = 100 # MVA
# sys.gfmi_e[0].i_load_ref = P_load/Sbase 
# sys.gfmi_e[0].i_load_ref = 0

# # sys.gfmi_e[0].SOC_init_pu = 0.0 
# # sys.gfmi_e[0].Pbat_max_pu = 100.0 
# # sys.gfmi_e[0].v_dc_ref = 1.00 


# # Run power flow
# pf = ACPowerFlow(system=sys, model_settings=None, solver_settings=None)
# pf.solve()
    
# # Break down lines into branches and shunts for small-signal modeling
# sys_modifier = SystemModifier(system=sys)
# sys_modifier.decompose_lines()
# sys_modifier.combine_shunts()

# # Construct small-signal model
# ssm = SmallSignalModel(system=sys)
# ssm.construct_system_ssm()

# ssm.simulate_ssm(t_max=t_max, inputs=inputs)

# emt_sc = SimulationEMT(system=sys)
# emt_sc.sim(t_max, inputs)

# # calculate SIL 

# # characteristic impedance
# import numpy as np 
# l = sys.lines[0]
# Vbase = l.base_voltage_kV*1e3
# Sbase = l.base_power_MVA*1e6
# wbase = 2*np.pi*60 
# Zbase = Vbase**2/Sbase 
# Ybase = 1/Zbase 
# l = sys.lines[0]
# x = l.x_pu*Zbase 
# r = l.r_pu*Zbase
# g = l.g_pu*Ybase 
# b = l.b_pu*Ybase  
# Z0 = np.sqrt((r + x*1j)/(g + b*1j))

# # for 230kV line, Z0 should be 360-390 Ohms...

# Z0_ideal = np.sqrt(x/b)
# SIL_ideal_MW = (Vbase**2/Z0_ideal)/1e6

# # Average transmission line parameter values
# # HV (69-230kV) - L=1.3mH/km, c=8.75nF/km 




print('ok')
# %%
