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
from sting.system.component import Component
from sting.generator.core import Generator
from sting.timescales.core import Timepoint
from sting.utils.data_tools import timeit

# Set up logging
logger = logging.getLogger(__name__)

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class EnergyBudget(Component):
    """Class representing an energy budget constraint over a set of generators and timepoints."""
    
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

        self.generators = [g for g in system.gens.to_list() if (g.name in region_gens)]
        self.timepoints = [t for t in system.timepoints if t.name in EnergyBudget._cached_terms[self.budget_term]]
