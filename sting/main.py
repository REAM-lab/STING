import os
import logging

from sting.system.core import System
from sting.utils.dynamical_systems import modal_analisis
from sting.utils.power_flow import PowerFlow
from sting.simulation_emt import SimulationEMT


def run_ssm(case_dir=os.getcwd(), write_outputs=True, log=True):

    #logging.basicConfig(filename="main.log", level=logging.INFO)

    # Set up grid from CSV files
    sys = System.from_csv(case_dir=case_dir)

    # Run AC-OPF
    sys.clean_up()
    pf = PowerFlow(system=sys)
    pf_instance = pf.run_acopf()

    # Use AC-OPF solution to build small-signal models
    sys.construct_ssm(pf_instance)
    ssm = sys.interconnect()

    # Analysis of final system stability
    modal_analisis(ssm.A, show=True)

    # Save the interconnected system model
    if write_outputs:
        path = os.path.join(case_dir, "outputs", "small_signal_model")
        ssm.to_csv(path)

    return sys, ssm

def run_emt(t_max, inputs, case_dir=os.getcwd()):

    sys, ssm = run_ssm(case_dir)

    sys.define_emt_variables()
    
    solution = sys.sim_emt(t_max, inputs)

    return solution, sys

def run_emt2(t_max, inputs, case_dir=os.getcwd()):

    sys, ssm = run_ssm(case_dir)

    emt_sc = SimulationEMT(system=sys, case_directory=case_dir)

    solution = emt_sc.sim(t_max, inputs)

    emt_sc.plot_results()

    return solution, sys