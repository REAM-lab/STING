# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar
import pyomo.environ as pyo
import os
import polars as pl
import logging
from collections import defaultdict

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
def construct_unit_commitment_model(system: System, model: pyo.ConcreteModel, model_settings: dict):
    """Construction of energy budget constraints."""
    
    S: list[Scenario] = system.scenarios
    B: list[EnergyBudget] = system.energy_budgets

    stochastic_gens_at_budget = defaultdict(list)
    nonstochastic_gens_at_budget = defaultdict(list)
    for b in B:
        for g in b.generators:
            if g.site != 'no_capacity_factor' and g.site is not None:
                stochastic_gens_at_budget[b].append(g)
            else:
                nonstochastic_gens_at_budget[b].append(g)
    
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
        return ( 0.01 * sum(m.vGEN_ST[g, s, t] for g in stochastic_gens_at_budget[b]) 
                + 0.01 * sum(m.vGEN_NST[g, t] for g in nonstochastic_gens_at_budget[b]) <= (b.budget_constraint_power_GW*10) )
    
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
        rule=lambda m, b, s: 1e-2 * sum(m.vGEN_ST[g, s, t] * t.weight for g in stochastic_gens_at_budget[b] for t in b.timepoints) 
                            + 1e-2 * sum(m.vGEN_NST[g, t] * t.weight for g in nonstochastic_gens_at_budget[b] for t in b.timepoints) == 10 * m.vAUX_ENERGY_BUDGET[s, b])
    # Enforce that total energy dispatched in GWh is less than budget (using 1e-2 to scale large RHS values)
    model.cEnergyBudget = pyo.Constraint(
        B, S, rule=lambda m, b, s: 1e-2 * m.vAUX_ENERGY_BUDGET[s, b] <= 1e-2 * b.budget_constraint_energy_GWh)

    logger.info(f"   Size: {len(model.cEnergyBudget)} constraints")
