# In progress - March 5, 2026 - Ruth 

# Import Python standard and third-party packages
from pathlib import Path

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
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

def step3(t):
    return 0.1 if t > 1.0 else 0.0 

# Specify inputs to excite - any constant input does not need to be specified 
# NB: input is a perturbation from the nominal value 
inputs = {'infinite_sources_0': {'v_ref_d': step1}, 
          'gfmi_e_0': {'p_ref': step2, 
                       'q_ref': step2,
                       'v_ref': step2,
                       'v_dc_ref': step2,
                       'v_s': step2, 
                       'i_load_ref': step2}}

t_max = 4.0

# Load system from CSV files
sys = System.from_csv(case_directory=case_dir)

# Match power reference with load 
P_load = 10 # MW 
sys.gfmi_e[0].minimum_active_power_MW = -P_load
sys.gfmi_e[0].maximum_active_power_MW = -P_load
sys.gfmi_e[0].minimum_reactive_power_MVAR = 0
sys.gfmi_e[0].maximum_reactive_power_MVAR = 0

# i_load_ref*v_dc_ref = i_load_ref*1 = p_load = i_load_ref  
Sbase = 100 # MVA
sys.gfmi_e[0].i_load_ref = P_load/Sbase 

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

ssm.simulate_ssm(t_max=t_max, inputs=inputs)

emt_sc = SimulationEMT(system=sys)
emt_sc.sim(t_max, inputs)

print('ok')