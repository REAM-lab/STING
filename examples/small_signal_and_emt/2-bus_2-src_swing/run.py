"""
Simulates two sources with swing dynamics connected via a transmission line.

First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.

You should obtain the following eigenvalues:
shape: (14, 5)
┌──────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real     ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---      ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64      ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ 0.0      ┆ 0.0       ┆ 0.0                  ┆ -1.0             ┆ -2.6276e7             │
│ -0.063   ┆ 8.892     ┆ 1.415                ┆ 0.007            ┆ 15.8731               │
│ -0.063   ┆ -8.892    ┆ 1.415                ┆ 0.007            ┆ 15.8731               │
│ -0.163   ┆ 0.0       ┆ 0.026                ┆ 1.0              ┆ 6.1304                │
│ -21.987  ┆ 376.796   ┆ 60.071               ┆ 0.058            ┆ 0.0455                │
│ -21.987  ┆ -376.796  ┆ 60.071               ┆ 0.058            ┆ 0.0455                │
│ -159.587 ┆ 2149.013  ┆ 342.968              ┆ 0.074            ┆ 0.0063                │
│ -159.587 ┆ -2149.013 ┆ 342.968              ┆ 0.074            ┆ 0.0063                │
│ -159.589 ┆ 2902.987  ┆ 462.722              ┆ 0.055            ┆ 0.0063                │
│ -159.589 ┆ -2902.987 ┆ 462.722              ┆ 0.055            ┆ 0.0063                │
│ -166.836 ┆ 3750.796  ┆ 597.548              ┆ 0.044            ┆ 0.006                 │
│ -166.836 ┆ -3750.796 ┆ 597.548              ┆ 0.044            ┆ 0.006                 │
│ -166.837 ┆ 4504.775  ┆ 717.449              ┆ 0.037            ┆ 0.006                 │
│ -166.837 ┆ -4504.775 ┆ 717.449              ┆ 0.037            ┆ 0.006                 │
└──────────┴───────────┴──────────────────────┴──────────────────┴───────────────────────┘
"""

# Import Python standard and third-party packages
from pathlib import Path
# Import sting package
from sting import main
from sting.system.core import System
import os
import numpy as np
import polars as pl

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Step function inputs to simulate
def step1(t):
    return 0.01 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

inputs = {
    'sources_with_swing_0': {
        'v_ref_d': step1
        }, 
    'sources_with_swing_1': {
        'v_ref_d': step2
        }
    }
t_max = 1.0 # Simulation length in seconds

# Construct system and small-signal model
sys, ssm = main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

# Run EMT simulation
main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

emt_dir = os.path.join(case_dir, "outputs", "simulation_emt")
ssm_dir = os.path.join(case_dir, "outputs", "small_signal_model")

ans = dict()
for component in sys:
    if hasattr(component, "compare_ssm_emt"):
        ans |= getattr(component, "compare_ssm_emt")(emt_dir, ssm_dir)
        
mprint("\nok")