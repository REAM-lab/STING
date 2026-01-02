# ----------------------
# Import python packages
# ----------------------
import os
import logging

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.modules.power_flow import PowerFlow
from sting.modules.simulation_emt import SimulationEMT
from sting.modules.small_signal_modeling import SmallSignalModel

# ----------------
# Main functions
# ----------------
def run_acopf(case_directory = os.getcwd()):
    """
    Routine to run AC optimal power flow from a case study directory.
    """
    # Load system from CSV files
    sys = System.from_csv(case_directory=case_directory)

    # Run power flow
    pf = PowerFlow(system=sys)
    pf.run_acopf()

    return sys

def run_ssm(case_directory = os.getcwd()):
    """
    Routine to construct the system and its small-signal model from a case study directory.
    """
    # Load system from CSV files
    sys = System.from_csv(case_directory=case_directory)

    # Run power flow
    pf = PowerFlow(system=sys)
    pf.run_acopf()

    # Construct small-signal model
    ssm = SmallSignalModel(system=sys)
    ssm.construct_system_ssm()

    return sys, ssm

def run_emt(t_max, inputs, case_directory=os.getcwd()):

    # Load system from CSV files
    sys = System.from_csv(case_directory=case_directory)

    # Run power flow
    pf = PowerFlow(system=sys)
    pf.run_acopf()

    # Construct small-signal model
    ssm = SmallSignalModel(system=sys)
    ssm.construct_system_ssm()

    emt_sc = SimulationEMT(system=sys)
    emt_sc.sim(t_max, inputs)

    return sys