# ----------------------
# Import python packages
# ----------------------
import os
import logging
import time

logging.basicConfig(level=logging.INFO,
                        format='%(message)s')

logger = logging.getLogger(__name__)


# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.modules.power_flow import PowerFlow
from sting.modules.simulation_emt import SimulationEMT
from sting.modules.small_signal_modeling import SmallSignalModel
from sting.modules.capacity_expansion import CapacityExpansion
from sting.modules.kron_reduction import KronReduction
from sting.utils.data_tools import setup_logging_file


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
    """
    Routine to simulate the EMT dynamics of the system from a case study directory.
    """

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



def run_capex(case_directory=os.getcwd(), model_settings=None, solver_settings=None):
    """
    Routine to perform capacity expansion analysis from a case study directory.
    """
    start_time = time.time()
    # Set up logging to file
    setup_logging_file(case_directory)

    # Load system from CSV files
    system = System.from_csv(case_directory=case_directory)
    
    # Perform capacity expansion analysis
    capex = CapacityExpansion(system=system , model_settings=model_settings, solver_settings=solver_settings)
    capex.solve()  
    logger.info(f"\n>> Run completed in {time.time() - start_time:.2f} seconds.\n")

    return system

def run_kron(case_directory=os.getcwd(), kron_settings=None, solver_settings=None):
    """
    Routine to perform Kron reduction from a case study directory.
    """

    start_time = time.time()
    # Set up logging to file
    setup_logging_file(case_directory)
    
    # Load system from CSV files
    system = System.from_csv(case_directory=case_directory)

    # Perform Kron reduction
    kr = KronReduction(original_system=system, settings=kron_settings, solver_settings=solver_settings)
    kr.reduce()

    logger.info(f"\n>> Run completed in {time.time() - start_time:.2f} seconds.\n")
    return kr

def run_kron_capex(case_directory=os.getcwd(), model_settings=None, solver_settings=None, kron_settings=None):
    """
    Perform capacity expansion analysis with Kron reduction from a case study directory.
    """
    
    # Set up logging to file
    setup_logging_file(case_directory)

    start_time = time.time()
    # Load system from CSV files
    system = System.from_csv(case_directory=case_directory)
    
    # Perform Kron reduction
    kr = KronReduction(original_system=system, settings=kron_settings, solver_settings=solver_settings)
    kr.reduce()
    
    output_directory = os.path.join(case_directory, "outputs", "kron_capacity_expansion")
    # Perform capacity expansion analysis
    capex = CapacityExpansion(
        system=kr.kron_system, 
        kron_variables=kr.to_variables(),
        model_settings=model_settings, 
        solver_settings=solver_settings, 
        output_directory=output_directory)
    capex.solve()  

    logger.info(f"\n>> Run completed in {time.time() - start_time:.2f} seconds.\n")
    return capex, kr