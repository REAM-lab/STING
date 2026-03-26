"""
Simulates a infinite source and GFMI connected via a multi-
segment pi transmission model.

        ┌──┬────www───uuu────┬──┐ ┌──┬────www───uuu────┬──┐
lima ═╪═╡                       ╞═╡                       ╞═╪═ santiago
GRID ─┘                        chile                      └─ GFMI

First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.

You should obtain the following eigenvalues:
┌───────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real      ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---       ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64       ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞═══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -4.322    ┆ 0.722     ┆ 0.697                ┆ 0.986            ┆ 0.2314                │
│ -4.322    ┆ -0.722    ┆ 0.697                ┆ 0.986            ┆ 0.2314                │
│ -19.41    ┆ 376.286   ┆ 59.967               ┆ 0.052            ┆ 0.0515                │
│ -19.41    ┆ -376.286  ┆ 59.967               ┆ 0.052            ┆ 0.0515                │
│ -65.887   ┆ 0.0       ┆ 10.486               ┆ 1.0              ┆ 0.0152                │
│ …         ┆ …         ┆ …                    ┆ …                ┆ …                     │
│ -1000.0   ┆ 0.0       ┆ 159.155              ┆ 1.0              ┆ 0.001                 │
│ -1559.822 ┆ 3703.372  ┆ 639.558              ┆ 0.388            ┆ 0.0006                │
│ -1559.822 ┆ -3703.372 ┆ 639.558              ┆ 0.388            ┆ 0.0006                │
│ -1636.806 ┆ 5409.28   ┆ 899.464              ┆ 0.29             ┆ 0.0006                │
│ -1636.806 ┆ -5409.28  ┆ 899.464              ┆ 0.29             ┆ 0.0006                │
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