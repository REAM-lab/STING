
# Import Python standard and third-party packages
from pathlib import Path
from scipy.linalg import eig, inv
import matplotlib.pyplot as plt
import sys 
import seaborn as sns 
import pandas as pd 
import numpy as np 
import os 
from plotly.subplots import make_subplots
import plotly.graph_objects as go

sys.path.append("/Users/ruthkravis/Documents/STING")
# Import sting package
from sting import main
from sting.system.core import System
from scipy import signal 



# Step-change input to applied to the system
def step1(t):
    return 0.2 if t >= 0.2 else 0.0

def step2(t):
    return 0.0

def step3(t):
    return 0.1 if t >= 1.0 else 0.0


def sin_oscillation(t):
    return 0.05*np.sin(2*np.pi*20*t) if t < 1 else 0  #1 Hz oscillation

def square_oscillation(t):
    osc = 0.05*signal.square(2 * np.pi * 14 * t)
    return osc

inputs = {
    'infinite_sources_0': {
        'v_ref_d': step1
        }, 
    'gfli_e_0': {
        'i_load_ref': step2
        }
    }

t_max = 1.0 # Simulation length (in seconds)

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
sys = System.from_csv(case_directory=case_dir)

# Construct system and small-signal model
_, ssm =  main.run_ssm(case_directory=case_dir)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
main.run_emt(case_directory=case_dir, inputs=inputs, t_max=t_max)

print('\nok')