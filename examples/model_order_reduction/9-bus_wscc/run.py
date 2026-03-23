"""

"""

# Import Python standard and third-party packages
from pathlib import Path

# Import sting package
from sting import main
from sting.system.core import System

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
sys = System.from_csv(case_directory=case_dir)

# Construct system and small-signal model
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

inputs = {'infinite_sources_0': {'v_ref_d': step1}}

t_max = 2.0

_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('ok')