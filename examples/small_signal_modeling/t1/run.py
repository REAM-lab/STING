"""
Testcase1 simulates a two infinite sources connected via a transmission line.

First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.

You should obtain the following eigenvalues:
Modal analysis results:
shape: (10, 5)
┌──────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real     ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---      ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64      ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -21.965  ┆ 376.991   ┆ 60.102               ┆ 0.058            ┆ 0.0455                │
│ -21.965  ┆ -376.991  ┆ 60.102               ┆ 0.058            ┆ 0.0455                │
│ -159.588 ┆ 2149.008  ┆ 342.967              ┆ 0.074            ┆ 0.0063                │
│ -159.588 ┆ -2149.008 ┆ 342.967              ┆ 0.074            ┆ 0.0063                │
│ -159.588 ┆ 2902.99   ┆ 462.723              ┆ 0.055            ┆ 0.0063                │
│ -159.588 ┆ -2902.99  ┆ 462.723              ┆ 0.055            ┆ 0.0063                │
│ -166.837 ┆ 3750.794  ┆ 597.548              ┆ 0.044            ┆ 0.006                 │
│ -166.837 ┆ -3750.794 ┆ 597.548              ┆ 0.044            ┆ 0.006                 │
│ -166.837 ┆ 4504.776  ┆ 717.449              ┆ 0.037            ┆ 0.006                 │
│ -166.837 ┆ -4504.776 ┆ 717.449              ┆ 0.037            ┆ 0.006                 │
└──────────┴───────────┴──────────────────────┴──────────────────┴───────────────────────┘
"""

# Import Python standard and third-party packages
from pathlib import Path

# Import sting package
from sting import main
from sting.system.core import System

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent



# Run EMT simulation
# Construct system and small-signal model
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

inputs = {'infinite_sources_0': {'v_ref_d': step1}, 
          'infinite_sources_1': {'v_ref_d': step2}}

t_max = 1.0

_, ssm = main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('ok')