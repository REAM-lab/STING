import os
import logging

from sting.system.core import System
from sting.utils.dynamical_systems import modal_analisis
from sting.utils.power_flow import PowerFlow


def run_ssm(inputs_dir=None, write_outputs=True, log=True):

    logging.basicConfig(filename="main.log", level=logging.INFO)

    # Set up grid from CSV files
    sys = System.from_csv(inputs_dir=inputs_dir)

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
        path = os.path.join(os.getcwd(), "outputs", "small_signal_model")
        ssm.to_csv(path)

    return sys, ssm
