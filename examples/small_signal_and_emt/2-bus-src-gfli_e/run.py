
# Import Python standard and third-party packages
from pathlib import Path

# Import sting package
from sting import main

# Step-change input to applied to the system
def step1(t):
    return 0.2 if t >= 0.2 else 0.0

def step2(t):
    return 0.0

inputs = {
    'infinite_sources_0': {
        'v_ref_d': step1
        }, 
    'gfli_e_0': {
        'i_load_ref': step2
        }
    }

t_max = 1.0 # Simulation length (in seconds)

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('\nok')