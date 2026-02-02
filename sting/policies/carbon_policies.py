# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar
import pyomo.environ as pyo
import os
import polars as pl
import logging

# -------------
# Import sting code
# --------------
from sting.generator.generator import Generator
from sting.timescales.core import Timepoint
from sting.utils.data_tools import timeit

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class CarbonPolicy:
    """Class representing an carbon policy constraint. It applies for the full model horizon."""

    id: int = field(default=-1, init=False)
    carbon_cap_tonneCO2peryear: float

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id
    
@timeit
def construct_capacity_expansion_model(system, model: pyo.ConcreteModel, model_settings: dict):
    """Construction of energy budget constraints."""

    logger.info(" - Annual carbon policy constraint")
    def cAnnualCarbonCap_rule(m, carbon_policy):
        return  0.1 * sum(m.eEmissionsPerTp[t] * t.weight for t in system.tp) <= carbon_policy.carbon_cap_tonneCO2peryear * 0.1
        
    model.cAnnualCarbonCap = pyo.Constraint(system.carbon_policy, rule=cAnnualCarbonCap_rule)
    logger.info(f"   Size: {len(model.cAnnualCarbonCap)} constraints")

@timeit
def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):
    """Export energy budget results to CSV files."""

    df = pl.DataFrame( schema=['carbon_cap_tonneCO2peryear', 'total_emissions_tonneCO2peryear'],
                        data=map(lambda tuple: (tuple[0].carbon_cap_tonneCO2peryear, tuple[1]), 
                                                zip(model.cAnnualCarbonCap, pyo.value(model.cAnnualCarbonCap[:]))) )

    df.write_csv(os.path.join(output_directory, "annual_carbon_policy_constraints.csv"))  