# In progress - March 5, 2026 - Ruth 

# Import Python standard and third-party packages
from pathlib import Path

# Import sting package
from sting import main
from sting.system.core import System

from sting.system.component import Component
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
from sting.modules.simulation_emt.core import SimulationEMT

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
inputs = {'infinite_sources_0': {'v_ref_d': step3}, 
          'gfmi_e_0': {'p_ref': step2, 
                       'q_ref': step2,
                       'v_ref': step2,
                       'v_dc_ref': step2,
                       'v_s': step1, 
                       'i_load_ref': step2}}

t_max = 3.0

# Load system from CSV files
sys = System.from_csv(case_directory=case_dir)

# Run power flow
pf = ACPowerFlow(system=sys, model_settings=None, solver_settings=None)
pf.solve()
    
_, ssm = main.run_ssm(case_directory=case_dir)

ssm.simulate_ssm(t_max=t_max, inputs=inputs)

main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('ok')