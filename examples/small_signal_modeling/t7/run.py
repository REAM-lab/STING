"""
Simulates a 5-bus system.


You should obtain the following eigenvalues:
┌───────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real      ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---       ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64       ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞═══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -2.469    ┆ 0.0       ┆ 0.393                ┆ 1.0              ┆ 0.4051                │
│ -4.548    ┆ 0.0       ┆ 0.724                ┆ 1.0              ┆ 0.2199                │
│ -5.433    ┆ 0.0       ┆ 0.865                ┆ 1.0              ┆ 0.184                 │
│ -5.612    ┆ 0.0       ┆ 0.893                ┆ 1.0              ┆ 0.1782                │
│ -7.54     ┆ 376.991   ┆ 60.012               ┆ 0.02             ┆ 0.1326                │
│ …         ┆ …         ┆ …                    ┆ …                ┆ …                     │
│ -3094.17  ┆ -6270.694 ┆ 1112.896             ┆ 0.442            ┆ 0.0003                │
│ -3100.194 ┆ 4057.577  ┆ 812.706              ┆ 0.607            ┆ 0.0003                │
│ -3100.194 ┆ -4057.577 ┆ 812.706              ┆ 0.607            ┆ 0.0003                │
│ -3163.709 ┆ 4060.411  ┆ 819.238              ┆ 0.615            ┆ 0.0003                │
│ -3163.709 ┆ -4060.411 ┆ 819.238              ┆ 0.615            ┆ 0.0003                │
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

inputs = {'infinite_sources_0': {'v_ref_d': step2}, 
          'gfmi_c_0': {'p_ref': step1}}

t_max = 2.0

_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)


print('ok')