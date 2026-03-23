"""
Example case study using a modified version of the IEEE 5-bus system with 
three GFMs and one infinite source.

You should obtain the following eigenvalues:
shape (59, 5):
┌───────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real      ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---       ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64       ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞═══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -3.569    ┆ 0.031     ┆ 0.568                ┆ 1.0              ┆ 0.2802                │
│ -3.569    ┆ -0.031    ┆ 0.568                ┆ 1.0              ┆ 0.2802                │
│ -4.383    ┆ 0.0       ┆ 0.698                ┆ 1.0              ┆ 0.2282                │
│ -5.479    ┆ 0.0       ┆ 0.872                ┆ 1.0              ┆ 0.1825                │
│ -29.66    ┆ 376.638   ┆ 60.129               ┆ 0.079            ┆ 0.0337                │
│ …         ┆ …         ┆ …                    ┆ …                ┆ …                     │
│ -3164.296 ┆ -4078.776 ┆ 821.603              ┆ 0.613            ┆ 0.0003                │
│ -3776.17  ┆ 8142.466  ┆ 1428.492             ┆ 0.421            ┆ 0.0003                │
│ -3776.17  ┆ -8142.466 ┆ 1428.492             ┆ 0.421            ┆ 0.0003                │
│ -3814.089 ┆ 5081.508  ┆ 1011.216             ┆ 0.6              ┆ 0.0003                │
│ -3814.089 ┆ -5081.508 ┆ 1011.216             ┆ 0.6              ┆ 0.0003                │
└───────────┴───────────┴──────────────────────┴──────────────────┴───────────────────────┘
"""

# Import Python standard and third-party packages
from pathlib import Path

# Import sting package
from sting import main
from sting.system.core import System

# Step-change input to applied to the system
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

inputs = {
    'infinite_sources_0': {
        'v_ref_d': step2
        }, 
    'gfmi_c_0': {
        'p_ref': step1
        }
    }

t_max = 2.0 # Simulation length (in seconds)

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
sys = System.from_csv(case_directory=case_dir)

# Construct system and small-signal model
_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation
main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('\nok')