"""
Simulates an infinite source and GFMI_c connected via a transmission line.

You should obtain the following eigenvalues:
shape: (19, 5)
┌───────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real      ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---       ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64       ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞═══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -4.569    ┆ 0.0       ┆ 0.727                ┆ 1.0              ┆ 0.2189                │
│ -4.588    ┆ 0.0       ┆ 0.73                 ┆ 1.0              ┆ 0.2179                │
│ -19.29    ┆ 376.581   ┆ 60.013               ┆ 0.051            ┆ 0.0518                │
│ -19.29    ┆ -376.581  ┆ 60.013               ┆ 0.051            ┆ 0.0518                │
│ -65.301   ┆ 0.0       ┆ 10.393               ┆ 1.0              ┆ 0.0153                │
│ …         ┆ …         ┆ …                    ┆ …                ┆ …                     │
│ -1000.06  ┆ 0.0       ┆ 159.164              ┆ 1.0              ┆ 0.001                 │
│ -1262.513 ┆ 4405.981  ┆ 729.454              ┆ 0.275            ┆ 0.0008                │
│ -1262.513 ┆ -4405.981 ┆ 729.454              ┆ 0.275            ┆ 0.0008                │
│ -1662.861 ┆ 5939.933  ┆ 981.715              ┆ 0.27             ┆ 0.0006                │
│ -1662.861 ┆ -5939.933 ┆ 981.715              ┆ 0.27             ┆ 0.0006                │
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
    'gfmi_c_0': {
        'p_ref': step1}
    }
t_max = 2.0 # Simulation length

# Construct system and small-signal model
_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation
main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('\nok')