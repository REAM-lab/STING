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
from sting.policies.transmission_expansion_constraint.core import TransmissionExpansionConstraint
from sting.utils.runtime_tools import timeit
from sting.modules.capacity_expansion.utils import ModelSettings

# Set up logging
logger = logging.getLogger(__name__)

@timeit
def construct_capacity_expansion_model(system: System, model: pyo.ConcreteModel, model_settings: ModelSettings):
    """Construction of constraint on transmission capacity expansion."""

    if model_settings.line_capacity_expansion and hasattr(model, "vCAPL"):

        if len(system.transmission_expansion_constraints) > 0:
        
            logger.info(" - Constraint on transmission capacity expansion")
            L_expandable = {l for l in system.lines if (l.expand_capacity == True and l.cap_existing_power_MW is not None)}
            def cTransmissionExpCapacity_rule(m: pyo.ConcreteModel, tx: TransmissionExpansionConstraint):
                return  0.01 * sum(m.vCAPL[l] for l in L_expandable) <= tx.built_transmission_capacity_cap_MW * 0.01
                
            model.cTransmissionExpCapacity = pyo.Constraint(system.transmission_expansion_constraints, rule=cTransmissionExpCapacity_rule)
            logger.info(f"   Size: {len(model.cTransmissionExpCapacity)} constraints")

        else:
            return
    
    else:

        if len(system.transmission_expansion_constraints) > 0:
            logger.info("Transmission expansion capacity policy is enabled but the model does not have line capacity expansion variables.")

@timeit
def export_results_capacity_expansion(system: System, model: pyo.ConcreteModel, output_directory: str):
    """Export transmission capacity expansion results to CSV files."""

    if hasattr(model, "cTransmissionExpCapacity") and len(model.cTransmissionExpCapacity) > 0:
        transmission_capacity_file = os.path.join(output_directory, "transmission_capacity_expansion_constraints.csv")
        (pl.DataFrame(
            data=( 
                (
                tx.id, 
                100 * pyo.value(model.cTransmissionExpCapacity[tx]),
                tx.built_transmission_capacity_cap_MW
                )
                    for tx in model.cTransmissionExpCapacity),
            schema=[
                "transmission_policy_id", 
                "built_transmission_capacity_MW",
                "built_transmission_capacity_cap_MW"
            ],
            orient="row")
            .write_csv(transmission_capacity_file))