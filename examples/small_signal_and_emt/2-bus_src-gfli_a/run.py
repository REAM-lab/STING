"""
Simulates an infinite source and GFLI_a connected via a transmission line.

You should obtain the following eigenvalues:
shape: (19, 5)
┌───────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real      ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---       ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64       ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞═══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -4.891    ┆ 0.312     ┆ 0.78                 ┆ 0.998            ┆ 0.2044                │
│ -4.891    ┆ -0.312    ┆ 0.78                 ┆ 0.998            ┆ 0.2044                │
│ -8.212    ┆ 0.0       ┆ 1.307                ┆ 1.0              ┆ 0.1218                │
│ -109.532  ┆ 189.808   ┆ 34.878               ┆ 0.5              ┆ 0.0091                │
│ -109.532  ┆ -189.808  ┆ 34.878               ┆ 0.5              ┆ 0.0091                │
│ …         ┆ …         ┆ …                    ┆ …                ┆ …                     │
│ -709.809  ┆ 5893.666  ┆ 944.784              ┆ 0.12             ┆ 0.0014                │
│ -709.809  ┆ -5893.666 ┆ 944.784              ┆ 0.12             ┆ 0.0014                │
│ -1430.224 ┆ 1052.832  ┆ 282.651              ┆ 0.805            ┆ 0.0007                │
│ -1430.224 ┆ -1052.832 ┆ 282.651              ┆ 0.805            ┆ 0.0007                │
│ -1537.006 ┆ 0.0       ┆ 244.622              ┆ 1.0              ┆ 0.0007                │
└───────────┴───────────┴──────────────────────┴──────────────────┴───────────────────────┘
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

# Step function to simulate
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

inputs = {
    'infinite_sources_0': {
        'v_ref_d': step2
        }, 
    'gfli_a_0': {
        'i_bus_d_ref': step1}
    }
t_max = 2.0 # Simulation length

# Construct system and small-signal model
_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation (not implemented yet for gfli_a)
# main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('\nok')