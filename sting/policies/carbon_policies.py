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
    def cAnnualCarbonCap_rule(m, carbon_policy, scenario):
        return  0.1 * sum(m.eEmissionsPerScPerTp[scenario, t] * t.weight for t in system.tp) <= carbon_policy.carbon_cap_tonneCO2peryear * 0.1
        
    model.cAnnualCarbonCap = pyo.Constraint(system.carbon_policy, system.sc, rule=cAnnualCarbonCap_rule)
    logger.info(f"   Size: {len(model.cAnnualCarbonCap)} constraints")