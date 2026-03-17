"""
Testcase1 simulates a two infinite sources connected via *TWO* transmission lines
in parallel. This can be used for testing network modifications like `combine_shunts()`

First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.

You should obtain the following eigenvalues:
┌───────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real      ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---       ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64       ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞═══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -4.743    ┆ 0.0       ┆ 0.755                ┆ 1.0              ┆ 0.2108                │
│ -6.271    ┆ 0.0       ┆ 0.998                ┆ 1.0              ┆ 0.1595                │
│ -7.54     ┆ 376.991   ┆ 60.012               ┆ 0.02             ┆ 0.1326                │
│ -7.54     ┆ -376.991  ┆ 60.012               ┆ 0.02             ┆ 0.1326                │
│ -24.193   ┆ 376.431   ┆ 60.034               ┆ 0.064            ┆ 0.0413                │
│ …         ┆ …         ┆ …                    ┆ …                ┆ …                     │
│ -1000.091 ┆ 0.0       ┆ 159.169              ┆ 1.0              ┆ 0.001                 │
│ -1672.173 ┆ 5730.831  ┆ 950.124              ┆ 0.28             ┆ 0.0006                │
│ -1672.173 ┆ -5730.831 ┆ 950.124              ┆ 0.28             ┆ 0.0006                │
│ -1672.38  ┆ 4109.784  ┆ 706.174              ┆ 0.377            ┆ 0.0006                │
│ -1672.38  ┆ -4109.784 ┆ 706.174              ┆ 0.377            ┆ 0.0006                │
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

# Run EMT simulation
# Construct system and small-signal model
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

inputs = {
    'infinite_sources_0': 
        {'v_ref_d': step2}, 
    'gfmi_c_0': 
        {'p_ref': step1}
    }

t_max = 2.0

_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('ok')