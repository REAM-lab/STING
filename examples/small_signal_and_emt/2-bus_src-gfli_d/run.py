"""
Simulates an infinite source and GFLI_d connected via a transmission line.

"""
# Import Python standard and third-party packages
from pathlib import Path

# Import sting package
from sting import main

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Step function to simulate
def step1(t):
    return 0.3 if t >= 0.1 else 0.0

def step2(t):
    return 0.0

inputs = {
    'infinite_sources_0': {
        'v_ref_d': step1
        }, 
    'gfli_d_0': {
        'i_load_ref': step2}
    }
t_max = 2.0 # Simulation length

# Construct system and small-signal model
_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation
main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('\nok')