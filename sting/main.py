# ----------------------
# Import python packages
# ----------------------
import os
import logging
import time

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
from sting.modules.simulation_emt.core import SimulationEMT
from sting.modules.small_signal_modeling.core import SmallSignalModel
from sting.modules.small_signal_modeling.operations import GroupBy
from sting.modules.capacity_expansion.core import CapacityExpansion
from sting.modules.kron_reduction.core import KronReduction
from sting.utils.runtime_tools import setup_logging_file

logging.basicConfig(level=logging.INFO,
                        format='%(message)s')

logger = logging.getLogger(__name__)



# ----------------
# Main functions
# ----------------
def run_acopf(case_directory = os.getcwd(), model_settings=None, solver_settings=None):
    """
    Routine to run AC optimal power flow from a case study directory.
    """
    start_time = time.time()
    
    # Set up logging to file
    setup_logging_file(case_directory)

    # Load system from CSV files
    sys = System.from_csv(case_directory=case_directory)

    # Run power flow
    pf = ACPowerFlow(system=sys, model_settings=model_settings, solver_settings=solver_settings)
    pf.solve()

    logger.info(f"\n>> Run completed in {time.time() - start_time:.2f} seconds.\n")

    return sys

def run_ssm(case_directory = os.getcwd(), model_settings=None, solver_settings=None):
    """
    Routine to construct the system and its small-signal model from a case study directory.
    """
    start_time = time.time()

    # Set up logging to file
    setup_logging_file(case_directory)

    # Load system from CSV files
    sys = System.from_csv(case_directory=case_directory)

    # Run power flow
    pf = ACPowerFlow(system=sys, model_settings=model_settings, solver_settings=solver_settings)
    pf.solve()

    # Break down lines into branches and shunts for small-signal modeling
    sys_modifier = SystemModifier(system=sys)
    sys_modifier.decompose_lines()

    # Construct small-signal model
    ssm = SmallSignalModel(system=sys)
    ssm.construct_system_ssm()

    logger.info(f"\n>> Run completed in {time.time() - start_time:.2f} seconds.\n")

    return sys, ssm

def run_emt(t_max, inputs, case_directory=os.getcwd(), model_settings=None, solver_settings=None):
    """
    Routine to simulate the EMT dynamics of the system from a case study directory.
    """

    # Load system from CSV files
    sys = System.from_csv(case_directory=case_directory)

    # Run power flow
    pf = ACPowerFlow(system=sys, model_settings=model_settings, solver_settings=solver_settings)
    pf.solve()

    # Break down lines into branches and shunts for small-signal modeling
    sys_modifier = SystemModifier(system=sys)
    sys_modifier.decompose_lines()

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
    capex = CapacityExpansion(system=system, model_settings=model_settings, solver_settings=solver_settings)
    capex.solve()  
    logger.info(f"\n>> Run completed in {time.time() - start_time:.2f} seconds.\n")

    return capex, system

def run_kron(case_directory=os.getcwd(), kron_settings=None, solver_settings=None):
    """
    Function to perform Kron reduction from a case study directory.
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
    Function to run capacity expansion analysis with Kron reduction from a case study directory.
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

def run_zonal_capex(case_directory=os.getcwd(), model_settings: dict = None, solver_settings: dict = None, components_to_clone: list[str] = None):
    """
    Function to run capacity expansion analysis with manual zonal grouping from a case study directory.
    """
    
    # Set up logging to file
    setup_logging_file(case_directory)

    start_time = time.time()
    # Load system from CSV files
    system = System.from_csv(case_directory=case_directory)
    
    # Perform manual zonal grouping
    sys_modifier = SystemModifier(system=system)
    zonal_system = sys_modifier.group_by_zones(components_to_clone=components_to_clone)

    # Save zonal system to CSV files
    zonal_system.write_csv(types = [int, float, str, bool])
    
    output_directory = os.path.join(case_directory, "outputs", "zonal_capacity_expansion")
    # Perform capacity expansion analysis
    capex = CapacityExpansion(
        system=zonal_system, 
        model_settings=model_settings, 
        solver_settings=solver_settings, 
        output_directory=output_directory)
    capex.solve()  

    logger.info(f"\n>> Run completed in {time.time() - start_time:.2f} seconds.\n")
    return capex, zonal_system

def run_capex_with_initial_build(case_directory=os.getcwd(), model_settings=None, solver_settings=None,
                                 output_directory=None,
                                 built_capacity_directory=None, make_non_expandable=False):
    """
    Function to run capacity expansion analysis with initial built capacities from a previous solution. 
    """
    # Set up logging to file
    setup_logging_file(case_directory)

    start_time = time.time()

    # Load system from CSV files
    system = System.from_csv(case_directory=case_directory)

    # If built_capacity_directory is not provided, it will look for built capacities in the default output directory of the capacity expansion results.
    if built_capacity_directory is None:
        built_capacity_directory = os.path.join(case_directory, "outputs", "capacity_expansion")

    # Upload built capacities
    sys_modifier = SystemModifier(system=system)
    sys_modifier.upload_built_capacities_from_csv(built_capacity_directory=built_capacity_directory, 
                                            make_non_expandable=make_non_expandable)

    # Perform capacity expansion analysis
    capex = CapacityExpansion(system=system, model_settings=model_settings, solver_settings=solver_settings,
                              output_directory=output_directory)

    # Solve capacity expansion
    capex.solve()

    logger.info(f"\n>> Run completed in {time.time() - start_time:.2f} seconds.\n")

    return capex, system


def run_mor_setup(case_directory = os.getcwd(), model_settings=None, solver_settings=None):
    """
    Routine to construct the system and its small-signal model that can be used with model reduction methods.
    """
    # Set up logging to file
    setup_logging_file(case_directory)

    # Load system from CSV files
    sys = System.from_csv(case_directory=case_directory)

    # Run power flow
    pf = ACPowerFlow(system=sys, model_settings=model_settings, solver_settings=solver_settings)
    pf.solve()

    # Break down lines into branches and shunts for small-signal modeling
    sys_modifier = SystemModifier(system=sys)
    sys_modifier.decompose_lines()

    # Construct small-signal model
    ssm = SmallSignalModel(system=sys)
    ssm.construct_system_ssm()

    # Interconnect all components in the same zone
    zonal_ssm = GroupBy(ssm, "zone").interconnect()
    # Interconnect all zonal models
    zonal_ssm.construct_system_ssm(write_csv=False)

    # Manually write CSVs (to ensure non-conflicting paths)
    output_dir = os.path.join(zonal_ssm.output_directory, os.pardir)
    
    zonal_ssm.model.to_csv(
        filepath=os.path.join(output_dir, "zonal_small_signal_model"))
    zonal_ssm.write_csv_ccm_matrices(
        output_dir=os.path.join(output_dir, "zonal_component_connection_matrices"))

    return ssm, zonal_ssm