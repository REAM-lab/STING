"""
Testcase1 simulates a infinite source and GFMI connected via a multi-
segment pi transmission model.

        ┌──┬────WWW───uuu────┬──┐ ┌──┬────WWW───uuu────┬──┐
lima ═╪═╡                       ╞═╡                       ╞═╪═ santiago
  SG ─┘ │                      chile                      │ └─ GFMI
        └──┬─────────────────WWW───uuu─────────────────┬──┘
First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.

You should obtain the following eigenvalues:
┌───────────┬───────────┬──────────────────────┬──────────────────┬───────────────────────┐
│ real      ┆ imag      ┆ natural_frequency_hz ┆ damping_ratio_pu ┆ time_constant_seconds │
│ ---       ┆ ---       ┆ ---                  ┆ ---              ┆ ---                   │
│ f64       ┆ f64       ┆ f64                  ┆ f64              ┆ f64                   │
╞═══════════╪═══════════╪══════════════════════╪══════════════════╪═══════════════════════╡
│ -4.75     ┆ 0.0       ┆ 0.756                ┆ 1.0              ┆ 0.2105                │
│ -6.003    ┆ 0.0       ┆ 0.955                ┆ 1.0              ┆ 0.1666                │
│ -7.54     ┆ 376.991   ┆ 60.012               ┆ 0.02             ┆ 0.1326                │
│ -7.54     ┆ -376.991  ┆ 60.012               ┆ 0.02             ┆ 0.1326                │
│ -23.463   ┆ 376.443   ┆ 60.029               ┆ 0.062            ┆ 0.0426                │
│ …         ┆ …         ┆ …                    ┆ …                ┆ …                     │
│ -1000.075 ┆ 0.0       ┆ 159.167              ┆ 1.0              ┆ 0.001                 │
│ -1644.924 ┆ 4007.744  ┆ 689.488              ┆ 0.38             ┆ 0.0006                │
│ -1644.924 ┆ -4007.744 ┆ 689.488              ┆ 0.38             ┆ 0.0006                │
│ -1701.233 ┆ 5614.346  ┆ 933.672              ┆ 0.29             ┆ 0.0006                │
│ -1701.233 ┆ -5614.346 ┆ 933.672              ┆ 0.29             ┆ 0.0006                │
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