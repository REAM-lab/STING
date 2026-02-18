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
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
from sting.modules.simulation_emt.core import SimulationEMT
from sting.modules.small_signal_modeling.core import SmallSignalModel
from sting.modules.capacity_expansion.core import CapacityExpansion
from sting.modules.kron_reduction.core import KronReduction
from sting.utils.data_tools import setup_logging_file

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


def run_mor(case_directory = os.getcwd(), model_settings=None, solver_settings=None):
    """
    Routine to construct the system and its small-signal model from a case study directory.
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

    from sting.utils.data_tools import matrix_to_csv

    models = ssm.get_component_attribute("ssm")

    from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables, modal_analisis
    u_B = [str(i) for i in sum([x.u for x in models], DynamicalVariables(name=[])).to_list()]
    u_C = [str(i) for i in ssm.model.u.to_list()]

    y_B = [str(i) for i in sum([x.y for x in models], DynamicalVariables(name=[])).to_list()]
    y_C = [str(i) for i in ssm.model.y.to_list()]
    os.makedirs(os.path.join(case_directory, "outputs", "connection_matrices"), exist_ok=True)
    matrix_to_csv(matrix=ssm.F, filepath=os.path.join(case_directory, "outputs","connection_matrices", "F.csv"), index=u_B, columns=y_B)
    matrix_to_csv(matrix=ssm.G, filepath=os.path.join(case_directory,"outputs", "connection_matrices", "G.csv"), index=u_B, columns=u_C)
    matrix_to_csv(matrix=ssm.H, filepath=os.path.join(case_directory, "outputs","connection_matrices", "H.csv"), index=y_C, columns=y_B)
    matrix_to_csv(matrix=ssm.L, filepath=os.path.join(case_directory,"outputs", "connection_matrices", "L.csv"), index=y_C, columns=u_C)


    # new_ssm.construct_system_ssm()
    new_ssm = ssm.group_by("zone").interconnect()
    new_ssm = StateSpaceModel.from_interconnected(new_ssm.get_component_attribute("ssm"), new_ssm.ccm_matrices, u=lambda x: x[:4], y=lambda x: x)
    modal_analisis(new_ssm.A, show=True)

    os.makedirs(os.path.join(case_directory, "outputs","permuted_state_space_model"), exist_ok=True)
    new_ssm.to_csv(os.path.join(case_directory, "outputs", "permuted_state_space_model"))
    





    return new_ssm, ssm