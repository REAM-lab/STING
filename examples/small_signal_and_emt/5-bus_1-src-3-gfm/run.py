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
│ -1.078    ┆ 0.0       ┆ 0.172                ┆ 1.0              ┆ 0.928                 │
│ -3.602    ┆ 0.0       ┆ 0.573                ┆ 1.0              ┆ 0.2777                │
│ -4.304    ┆ 0.0       ┆ 0.685                ┆ 1.0              ┆ 0.2324                │
│ -5.467    ┆ 0.0       ┆ 0.87                 ┆ 1.0              ┆ 0.1829                │
│ -29.773   ┆ 376.392   ┆ 60.092               ┆ 0.079            ┆ 0.0336                │
│ …         ┆ …         ┆ …                    ┆ …                ┆ …                     │
│ -3168.151 ┆ -4078.546 ┆ 821.95               ┆ 0.613            ┆ 0.0003                │
│ -3773.037 ┆ 8140.806  ┆ 1428.042             ┆ 0.421            ┆ 0.0003                │
│ -3773.037 ┆ -8140.806 ┆ 1428.042             ┆ 0.421            ┆ 0.0003                │
│ -3817.213 ┆ 5081.402  ┆ 1011.502             ┆ 0.601            ┆ 0.0003                │
│ -3817.213 ┆ -5081.402 ┆ 1011.502             ┆ 0.601            ┆ 0.0003                │
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