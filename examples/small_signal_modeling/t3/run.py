# In progress - March 5, 2026 - Ruth 

# Import Python standard and third-party packages
from pathlib import Path

# Import sting package
from sting import main
from sting.system.core import System

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
def step1(t): 
    return 0.1 if t >= 0.5 else 0.0

def step2(t): # no perturbation  
    return 0.0

# Specify inputs to excite - any constant input does not need to be specified 
# NB: input is a perturbation from the nominal value 
inputs = {'infinite_sources_0': {'v_ref_d': step2}, 
          'gfmi_e_0': {'p_ref': step2, 
                       'q_ref': step2,
                       'v_ref': step2,
                       'v_dc_ref': step2,
                       'v_s': step2, 
                       'Pload': step2}}

t_max = 1.0

_, ssm = main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('ok')