# -------------
# Import python packages
# --------------
import os
import pyomo.environ as pyo
import polars as pl
import logging

# -------------
# Import sting code
# --------------
from sting.system.core import System
from sting.policies.carbon_policies.core import CarbonPolicy
from sting.timescales.core import Scenario, Timepoint
from sting.utils.runtime_tools import timeit
from sting.modules.capacity_expansion.utils import ModelSettings

# Set up logging
logger = logging.getLogger(__name__)

@timeit
def construct_capacity_expansion_model(system: System, model: pyo.ConcreteModel, model_settings: ModelSettings):
    """Construction of carbon policy constraints."""

    S: list[Scenario] = system.scenarios
    T: list[Timepoint] = system.timepoints
    # Intermediate variable for each carbon budget of the total emission per day scaled by 1e-2
    # Used to decouple the large RHS constraint coefficient (of total carbon budget)
    # from the small matrix coefficients in summing emissions. Additionally, the constraint now 
    # looks temporally sparse to the optimizer
    model.vAUX_CARBON_BUDGET = pyo.Var(S, T, within=pyo.NonNegativeReals)

    def cAuxCarbonCap_rule(m: pyo.ConcreteModel, scenario: Scenario, timepoint: Timepoint):
        return m.eEmissionsPerScPerTp[scenario, timepoint] == 1e-02 * m.vAUX_CARBON_BUDGET[scenario, timepoint]
    
    logger.info(" - Annual carbon policy constraint")
    def cAnnualCarbonCap_rule(m: pyo.ConcreteModel, carbon_policy: CarbonPolicy, scenario: Scenario):
        return  1e-03 * sum(m.vAUX_CARBON_BUDGET[scenario, t] * t.weight for t in system.timepoints) <= 1e-05 * carbon_policy.carbon_cap_tonneCO2peryear 
        
    model.cAuxCarbonCap = pyo.Constraint(S, T, rule=cAuxCarbonCap_rule)
    model.cAnnualCarbonCap = pyo.Constraint(system.carbon_policies, S, rule=cAnnualCarbonCap_rule)
    logger.info(f"   Size: {len(model.cAnnualCarbonCap)} constraints")


@timeit
def export_results_capacity_expansion(system: System, model: pyo.ConcreteModel, output_directory: str):
    """Export carbon policy results to CSV files."""
    
    if hasattr(model, "cAnnualCarbonCap"):
        carbon_policies_file = os.path.join(output_directory, "annual_carbon_cap_constraints.csv")
        (pl.DataFrame(
            data=((
                sc.name, 
                cp.id, 
                cp.carbon_cap_tonneCO2peryear,
                1e8 * pyo.value(model.cAnnualCarbonCap[cp, sc])) # To recover original units scale by: 1e8 = 1e5 * 1e3
                    for cp, sc in model.cAnnualCarbonCap),
            schema=[
                "scenario",
                "carbon_policy_id", 
                "carbon_cap_tonneCO2peryear",
                "dispatched_emissions_tonneCO2peryear"
            ],
            orient="row")
            .write_csv(carbon_policies_file))
