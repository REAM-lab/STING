# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar
import pyomo.environ as pyo
import os
import polars as pl

# -------------
# Import sting code
# --------------
from sting.generator.generator import Generator
from sting.timescales.core import Timepoint

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class EnergyBudget:
    """Class representing an energy budget constraint over a set of generators and timepoints."""
    
    id: int = field(default=-1, init=False)
    budget_region: str
    budget_term: str
    budget_energy_GWh: float
    generators: list[Generator] = None
    timepoints: list[Timepoint] = None

    # attributes shared across all instances
    _cache_initialized: ClassVar[bool] = False
    _cached_regions: ClassVar[dict] = None
    _cached_terms: ClassVar[dict] = None
    

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id
    
    def post_system_init(self, system):
        
        if EnergyBudget._cache_initialized is False:
            # Load timepoint to term mapping
            timepoint_to_term_df = pl.read_csv(os.path.join(system.case_directory, "inputs", "timepoint_to_term_budgets.csv"))
            terms_df = timepoint_to_term_df.group_by("budget_term").agg(pl.col("timepoint").alias("timepoints"))
            EnergyBudget._cached_terms = dict(zip(terms_df["budget_term"], terms_df["timepoints"]))

            # Load generator to region mapping
            generator_to_region_df = pl.read_csv(os.path.join(system.case_directory, "inputs", "generator_to_region_budgets.csv"))
            regions_df = generator_to_region_df.group_by("budget_region").agg(pl.col("generator").alias("generators"))
            EnergyBudget._cached_regions = dict(zip(regions_df["budget_region"], regions_df["generators"]))
            
            EnergyBudget._cache_initialized = True
    
        # Use cached data for this instance
        self.generators = [g for g in system.gen if g.name in EnergyBudget._cached_regions[self.budget_region]]
        self.timepoints = [t for t in system.tp if t.name in EnergyBudget._cached_terms[self.budget_term]]


def construct_capacity_expansion_model(system, model: pyo.ConcreteModel, model_settings: dict):

    def cEnergyBudget_rule(m, eb):
        return  sum(m.vGEN[g, t] * t.weight for g in eb.generators for t in eb.timepoints) * 1e-3 <= eb.budget_energy_GWh
        
    model.cEnergyBudget = pyo.Constraint(system.energy_budget, rule=cEnergyBudget_rule)


def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):
    """Export energy budget results to CSV file."""

    df = pl.DataFrame( schema=['budget_region', 'budget_term', 'budget_energy_GWh', "used_energy_GWh"],
                        data=map(lambda tuple: (tuple[0].budget_region, tuple[0].budget_term, tuple[0].budget_energy_GWh, tuple[1]), 
                                                zip(model.cEnergyBudget, pyo.value(model.cEnergyBudget[:]))) )

    df.write_csv(os.path.join(output_directory, "energy_budget_constraint.csv"))  