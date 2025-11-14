# import numpy as np
# from itertools import chain
from sting.system.core import System
from sting.utils import linear_systems_tools 
from sting.utils.power_flow import PowerFlow

def run_ssm():
    
    # Set up grid from CSV files
    sys = System.from_csv()
    
    # Run AC-OPF
    sys.prep_powerflow()
    pf = PowerFlow(system=sys)
    pf_instance = pf.run_acopf()

    # Use AC-OPF solution to build small-signal models
    sys.construct_ssm(pf_instance)
    ssm = sys.interconnect()

    # Analysis of final system stability
    linear_systems_tools.modal_analisis(ssm.A, show=True)
    
    return sys, ssm