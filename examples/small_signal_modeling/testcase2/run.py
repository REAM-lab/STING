"""
Testcase2 simulates a GFLI connected to an infinite source
via a transmission line.
"""

# Import Python standard and third-party packages
from pathlib import Path

# Import sting package
from sting import main

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
sys, ssm = main.run_ssm(case_dir)

print('ok')