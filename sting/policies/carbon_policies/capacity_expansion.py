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
from sting.timescales.core import Scenario
from sting.utils.data_tools import timeit
from sting.modules.capacity_expansion.utils import ModelSettings

# Set up logging
logger = logging.getLogger(__name__)

@timeit
def construct_capacity_expansion_model(system: System, model: pyo.ConcreteModel, model_settings: ModelSettings):
    """Construction of energy budget constraints."""

    logger.info(" - Annual carbon policy constraint")
    def cAnnualCarbonCap_rule(m: pyo.ConcreteModel, carbon_policy: CarbonPolicy, scenario: Scenario):
        return  0.01 * sum(m.eEmissionsPerScPerTp[scenario, t] * t.weight for t in system.timepoints) <= carbon_policy.carbon_cap_tonneCO2peryear * 0.01
        
    model.cAnnualCarbonCap = pyo.Constraint(system.carbon_policies, system.scenarios, rule=cAnnualCarbonCap_rule)
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
                100 * pyo.value(model.cAnnualCarbonCap[cp, sc]))
                    for cp, sc in model.cAnnualCarbonCap),
            schema=[
                "scenario",
                "carbon_policy_id", 
                "carbon_cap_tonneCO2peryear",
                "dispatched_emissions_tonneCO2peryear"
            ],
            orient="row")
            .write_csv(carbon_policies_file))
