"""
Simulates an infinite source and GFMI_c connected via a transmission line.

You should obtain the following eigenvalues:
shape: (19, 5)
┌───────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real      ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---       ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64       ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞═══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -4.321    ┆ 0.698     ┆ 0.697                ┆ 0.987            ┆ 0.2314                │
│ -4.321    ┆ -0.698    ┆ 0.697                ┆ 0.987            ┆ 0.2314                │
│ -19.414   ┆ 376.269   ┆ 59.965               ┆ 0.052            ┆ 0.0515                │
│ -19.414   ┆ -376.269  ┆ 59.965               ┆ 0.052            ┆ 0.0515                │
│ -65.905   ┆ 0.0       ┆ 10.489               ┆ 1.0              ┆ 0.0152                │
│ …         ┆ …         ┆ …                    ┆ …                ┆ …                     │
│ -1000.0   ┆ 0.0       ┆ 159.155              ┆ 1.0              ┆ 0.001                 │
│ -1271.429 ┆ 4404.988  ┆ 729.695              ┆ 0.277            ┆ 0.0008                │
│ -1271.429 ┆ -4404.988 ┆ 729.695              ┆ 0.277            ┆ 0.0008                │
│ -1653.37  ┆ 5937.579  ┆ 980.948              ┆ 0.268            ┆ 0.0006                │
│ -1653.37  ┆ -5937.579 ┆ 980.948              ┆ 0.268            ┆ 0.0006                │
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