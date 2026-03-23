"""
Example case study using a modified version of the WSCC IEEE 9-bus system with
three infinite source generators.

You should obtain the following eigenvalues:
shape: (42, 5)
┌──────────┬────────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real     ┆ imag       ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---      ┆ ---        ┆ ---                  ┆ ---              ┆ ---                   │
│ f64      ┆ f64        ┆ f64                  ┆ f64              ┆ f64                   │
╞══════════╪════════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -21.29   ┆ 376.991    ┆ 60.096               ┆ 0.056            ┆ 0.047                 │
│ -21.29   ┆ -376.991   ┆ 60.096               ┆ 0.056            ┆ 0.047                 │
│ -60.194  ┆ 376.991    ┆ 60.76                ┆ 0.158            ┆ 0.0166                │
│ -60.194  ┆ -376.991   ┆ 60.76                ┆ 0.158            ┆ 0.0166                │
│ -70.06   ┆ 376.991    ┆ 61.027               ┆ 0.183            ┆ 0.0143                │
│ …        ┆ …          ┆ …                    ┆ …                ┆ …                     │
│ -222.398 ┆ -13323.707 ┆ 2120.829             ┆ 0.017            ┆ 0.0045                │
│ -223.643 ┆ 6996.937   ┆ 1114.166             ┆ 0.032            ┆ 0.0045                │
│ -223.643 ┆ -6996.937  ┆ 1114.166             ┆ 0.032            ┆ 0.0045                │
│ -223.643 ┆ 7750.92    ┆ 1234.111             ┆ 0.029            ┆ 0.0045                │
│ -223.643 ┆ -7750.92   ┆ 1234.111             ┆ 0.029            ┆ 0.0045                │
└──────────┴────────────┴──────────────────────┴──────────────────┴───────────────────────┘
"""

# Import Python standard and third-party packages
from pathlib import Path
# Import sting package
from sting import main
from sting.system.core import System

# Input step function to simulate
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

inputs = {'infinite_sources_0': {'v_ref_d': step1}}
t_max = 2.0

# Create and simulate small-signal model and EMT dynamics
case_dir = Path(__file__).resolve().parent
sys = System.from_csv(case_directory=case_dir)

_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('\nok')