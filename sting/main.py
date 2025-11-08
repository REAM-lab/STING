import numpy as np
from itertools import chain

from sting.system.core import System
from sting.utils import linear_systems_tools 
from sting.utils.power_flow import Power_flow
from sting.systemic_small_signal_model import Composite_small_signal_model

def run_ssm():

    sys = System() 

    sys.load_components_via_input_csv_files()
    
    sys.dissect_lines_into_branches_and_shunts()

    pf = Power_flow(system=sys)

    pf_instance = pf.run_acopf()

    sys.transfer_power_flow_solution_to_components(pf_instance)

    sys.calculate_emt_initial_condition_of_components()

    sys.build_small_signal_model_of_components()

    sys_ssm = Composite_small_signal_model(input_system=sys)
    
    linear_systems_tools.modal_analisis(sys_ssm.system.A,
                                        show=True)
    
    return sys


