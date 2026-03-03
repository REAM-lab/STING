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
from sting.system.core import System
from sting.policies.energy_budgets.core import EnergyBudget
from sting.timescales.core import Timepoint, Scenario
from sting.utils.runtime_tools import timeit

# Set up logging
logger = logging.getLogger(__name__)

@timeit
def construct_capacity_expansion_model(system: System, model: pyo.ConcreteModel, model_settings: dict):
    """Construction of energy budget constraints."""
    
    S: list[Scenario] = system.scenarios
    B: list[EnergyBudget] = system.energy_budgets
    # Intermediate variable for each energy budget of the total dispatched energy in GWh
    # Used to decouple the large RHS constraint coefficient (of total allowable energy dispatch)
    # from the small matrix coefficients in summing power.  
    model.vAUX_ENERGY_BUDGET = pyo.Var(S, B, within=pyo.NonNegativeReals)

    # Constraints enforcing the maximum POWER that can be dispatched by a group of generators at
    # EACH timepoint in a budget term. These constraints are UNWEIGHTED.
    logger.info(" - Constraint for power budget")
    def cPowerBudget_rule(m: pyo.ConcreteModel, b: EnergyBudget, t: Timepoint, s: Scenario):
        #b, t = budget_tuple 
        # Converting GW to MW (but splitting the conversion over the both sides of the 
        # inequality for better numerical conditioning).
        return  0.01 * sum(m.vGEN[g, s, t] for g in b.generators) <= (b.budget_constraint_power_GW*10)
    
    # Flatten power budgets into a tuple for every timepoint
    power_budgets = []
    for b in system.energy_budgets:
        # Skip any budgets without a power constraint
        if b.budget_constraint_power_GW is None:
            continue
        for t in b.timepoints:
            power_budgets.append((b, t))

    model.cPowerBudget = pyo.Constraint(power_budgets, system.scenarios, rule=cPowerBudget_rule)
    logger.info(f"   Size: {len(model.cPowerBudget)} constraints")

    # Constraints enforcing the total ENERGY that can be dispatched by a group of generators
    # SUMMED over all timepoint in a budget term. These constraints are WEIGHTED.

    logger.info(" - Constraint for energy budget")
    # Enforce that the auxiliary variable equals total energy dispatched in GWh
    model.cAuxEnergyBudget = pyo.Constraint(
        B, S, 
        rule=lambda m, b, s: 1e-2 * sum(m.vGEN[g, s, t] * t.weight for g in b.generators for t in b.timepoints) == 10 * m.vAUX_ENERGY_BUDGET[s, b])
    # Enforce that total energy dispatched in GWh is less than budget (using 1e-2 to scale large RHS values)
    model.cEnergyBudget = pyo.Constraint(
        B, S, rule=lambda m, b, s: 1e-2 * m.vAUX_ENERGY_BUDGET[s, b] <= 1e-2 * b.budget_constraint_energy_GWh)

    logger.info(f"   Size: {len(model.cEnergyBudget)} constraints")

@timeit
def export_results_capacity_expansion(system: System, model: pyo.ConcreteModel, output_directory: str):
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
                1e2 * pyo.value(model.cEnergyBudget[b, sc]))
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