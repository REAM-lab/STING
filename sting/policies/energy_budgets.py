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

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class EnergyBudget:
    """Class representing an energy budget constraint over a set of generators and timepoints."""

    id: int = field(default=-1, init=False)
    budget_region: str
    budget_term: str
    budget_constraint_energy_GWh: float = None
    budget_constraint_power_GW: float = None
    generators: list[Generator] = None
    timepoints: list[Timepoint] = None

    # attributes shared across all instances
    _cache_initialized: ClassVar[bool] = False
    _cached_regions: ClassVar[dict] = None
    _cached_terms: ClassVar[dict] = None
    

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id
    
    def __repr__(self):
        return f"EnergyBudget(id={self.id}, budget_region='{self.budget_region}', budget_term='{self.budget_term}', budget_constraint_energy_GWh={self.budget_constraint_energy_GWh})"
    
    def post_system_init(self, system):
        
        if EnergyBudget._cache_initialized is False:
            # Load timepoint to term mapping
            timepoint_to_term_df = pl.read_csv(os.path.join(system.case_directory, "inputs", "energy_budget_terms.csv"))
            terms_df = timepoint_to_term_df.group_by("budget_term").agg(pl.col("timepoint").alias("timepoints"))
            EnergyBudget._cached_terms = dict(zip(terms_df["budget_term"], terms_df["timepoints"]))

            # Load generator to region mapping
            generator_to_region_df = pl.read_csv(os.path.join(system.case_directory, "inputs", "energy_budget_regions.csv"))
            regions_df = generator_to_region_df.group_by("budget_region").agg(pl.col("generator").alias("generators"))
            EnergyBudget._cached_regions = dict(zip(regions_df["budget_region"], regions_df["generators"]))

            # Mapping of the key pair (budget_region, generator.name) -> power_to_constraint
            EnergyBudget._cache_initialized = True
    
        # Use cached data for this instance
        region_gens = EnergyBudget._cached_regions[self.budget_region]

        self.generators = [g for g in system.gen if (g.name in region_gens)]
        self.timepoints = [t for t in system.tp if t.name in EnergyBudget._cached_terms[self.budget_term]]

@timeit
def construct_capacity_expansion_model(system, model: pyo.ConcreteModel, model_settings: dict):
    """Construction of energy budget constraints."""
    
    # Constraints enforcing the maximum POWER that can be dispatched by a group of generators at
    # EACH timepoint in a budget term. These constraints are UNWEIGHTED.
    logger.info(" - Constraint for power budget")
    def cPowerBudget_rule(m, b, t, s):
        #b, t = budget_tuple 
        # Converting GW to MW (but splitting the conversion over the both sides of the 
        # inequality for better numerical conditioning).
        return  0.01 * sum(m.vGEN[g, s, t] for g in b.generators) <= (b.budget_constraint_power_GW*10)
    
    # Flatten power budgets into a tuple for every timepoint
    power_budgets = []
    for b in system.energy_budget:
        # Skip any budgets without a power constraint
        if b.budget_constraint_power_GW is None:
            continue
        for t in b.timepoints:
            power_budgets.append((b, t))

    model.cPowerBudget = pyo.Constraint(power_budgets, system.sc, rule=cPowerBudget_rule)
    logger.info(f"   Size: {len(model.cPowerBudget)} constraints")

    # Constraints enforcing the total ENERGY that can be dispatched by a group of generators
    # SUMMED over all timepoint in a budget term. These constraints are WEIGHTED.
    logger.info(" - Constraint for energy budget")
    def cEnergyBudget_rule(m, b, s):
        return  0.01 * sum(m.vGEN[g, s, t] * t.weight for g in b.generators for t in b.timepoints) <= (b.budget_constraint_energy_GWh * 10)

    energy_budgets = [b for b in system.energy_budget if (b.budget_constraint_energy_GWh is not None)]  
    model.cEnergyBudget = pyo.Constraint(energy_budgets, system.sc, rule=cEnergyBudget_rule)

    logger.info(f"   Size: {len(model.cEnergyBudget)} constraints")

@timeit
def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):
    """Export energy budget results to CSV files."""
    # Write power budget constraints to CSV
    if hasattr(model, "cPowerBudget"):
        power_budgets_file = os.path.join(output_directory, "power_budget_constraints.csv")
        (pl.DataFrame(
            data=((
                sc.name, 
                b.budget_region, 
                b.budget_term, 
                t.name, 
                b.budget_constraint_power_GW, 
                1e-1 * pyo.value(model.cPowerBudget[b, t, sc])) 
                    for b, t, sc in model.cPowerBudget),
            schema=[
                "scenario",
                "budget_region", 
                "budget_term", 
                "timepoint",
                "budget_constraint_power_GW",
                "dispatched_power_GW"
            ],
            orient="row")
        .write_csv(power_budgets_file))
    
    # Write energy budget constraint to CSV
    if hasattr(model, "cEnergyBudget"):
        energy_budgets_file = os.path.join(output_directory, "energy_budget_constraints.csv")
        (pl.DataFrame(
            data=((
                sc.name, 
                b.budget_region, 
                b.budget_term, 
                b.budget_constraint_energy_GWh, 
                1e-1 * pyo.value(model.cEnergyBudget[b, sc]))
                    for b, sc in model.cEnergyBudget),
            schema=[
                "scenario",
                "budget_region", 
                "budget_term", 
                "budget_constraint_energy_GWh",
                "dispatched_energy_GWh"
            ],
            orient="row")
            .write_csv(energy_budgets_file))

    # [!] WARNING [!] The `_cache_initialized` attribute is GLOBAL to ALL EnergyBudget instances.
    # In order to run more than one capacity expansion model in the same python session we need to 
    # set this attribute back to False (so that new input datasets will be read).
    # Note: This is somewhat dangerous behavior and should be fixed in later updates.
    EnergyBudget._cache_initialized = False