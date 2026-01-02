# Import Python standard and third-party packages
import os
import matlab.engine
import numpy as np
import pandas as pd
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
from pathlib import Path

# Import sting package
from sting import main
from sting.utils.transformations import dq02abc, abc2dq0

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent
#case_dir = os.path.join(os.getcwd(), "examples", "testcase1")

# Construct system and small-signal model
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

inputs = {'inf_src_1': {'v_ref_d': step1}, 
          'inf_src_2': {'v_ref_d': step2}}

t_max = 1.0

main.run_emt(t_max, inputs, case_directory=case_dir)

print("Simulation completed.")