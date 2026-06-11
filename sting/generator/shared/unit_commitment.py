# ---------------
# Import standard libraries
# ---------------
import os
import logging
import polars as pl
import pyomo.environ as pyo
from collections import defaultdict

# ---------------
# Import sting code
# ---------------
from sting.bus.core import Bus
from sting.generator.core import Generator, CapacityFactor
from sting.timescales.core import Timepoint, Scenario
from sting.system.core import System
from sting.utils.runtime_tools import timeit
from sting.modules.capacity_expansion.utils import ModelSettings

# Set up logging
logger = logging.getLogger(__name__)

@timeit
def construct_unit_commitment_model(system: System, model: pyo.ConcreteModel, model_settings: ModelSettings):
    """Construction of generator variables, constraints, and costs."""

    S: list[Scenario] = system.scenarios
    T: list[Timepoint] = system.timepoints
    G: list[Generator] = system.gens.to_list()
    cf: list[CapacityFactor] = system.capacity_factors
    N: list[Bus] = system.buses

    # Classify generators into those with and without capacity factors, i.e., variable vs. non-variable generators
    NSG = [g for g in G if g.site == 'no_capacity_factor' or g.site is None]
    SG = [g for g in G if g.site != 'no_capacity_factor' and g.site is not None]

    logger.info(" - Decision variables of non-stochastic dispatch, i.e., dispatch with no capacity factor dependence")
    model.vGEN_NST = pyo.Var(NSG, T, within=pyo.NonNegativeReals)
    logger.info(f"   Size: {len(model.vGEN_NST)} variables")

    model.vGEN_DELTA = pyo.Var(NSG, S, T, within=pyo.Reals)

    logger.info(" - Decision variables of stochastic dispatch, i.e., dispatch dependent on capacity factors")
    model.vGEN_ST = pyo.Var(SG, S, T, within=pyo.NonNegativeReals)
    logger.info(f"   Size: {len(model.vGEN_ST)} variables")

    def generation_rule(m: pyo.ConcreteModel, g: Generator, s: Scenario, t: Timepoint):
        if g in NSG:
            return m.vGEN_NST[g, t] + m.vGEN_DELTA[g, s, t]
        else:
            return m.vGEN_ST[g, s, t]
    model.vGEN = pyo.Expression(G, S, T, rule=generation_rule)

    logger.info(" - Constraints on dispatch for non-variable generators (capacity factor of 1)")
    def max_dispatch_no_cf_rule(m: pyo.ConcreteModel, g: Generator, s: Scenario, t: Timepoint):
        return (0, m.vGEN[g, s, t], g.cap_existing_power_MW)
    model.cMaxDispatchNoCf = pyo.Constraint(NSG, S, T, rule=max_dispatch_no_cf_rule)
    logger.info(f"   Size: {len(model.cMaxDispatchNoCf)} constraints")

    logger.info(" - Constraints on dispatch for variable generators")
    cf_lookup = {(cf_inst.site, cf_inst.scenario, cf_inst.timepoint): cf_inst.capacity_factor for cf_inst in cf}

    def max_dispatch_rule(m: pyo.ConcreteModel, g: Generator, s: Scenario, t: Timepoint):

        # Generator capacity factor and nameplate power capacity
        capacity_factor = cf_lookup.get((g.site, s.name, t.name), None)
        if capacity_factor is None:
                logger.info(f"The site {g.site} for generator {g.name} does not have a corresponding capacity factor for scenario {s.name} and timepoint {t.name}. Check capacity_factors.csv")
                raise ValueError(f"Check capacity_factors.csv")

        x = (g.cap_existing_power_MW * capacity_factor)
        if x <= 1:
            scalefactor = 100
        elif x <= 10:
            scalefactor = 10
        elif x <= 100:
            scalefactor = 1
        else:
            scalefactor = 0.1

        return (0, scalefactor * m.vGEN[g, s, t], scalefactor * capacity_factor *  g.cap_existing_power_MW)

    model.cMaxDispatchCf = pyo.Constraint(SG, S, T, rule=max_dispatch_rule)
    logger.info(f"   Size: {len(model.cMaxDispatchCf)} constraints")

    logger.info(" - Expressions for dispatch at any bus")
    G_at_bus = defaultdict(list)
    for g in G:
        G_at_bus[g.bus_id].append(g)
    model.eGenAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: sum(m.vGEN[g, s, t] for g in G_at_bus[n.id]))
    logger.info(f"   Size: {len(model.eGenAtBus)} expressions")

    logger.info(" - Expressions for emission per timepoint per scenario expression")
    G_with_emissions = [g for g in G if g.emission_rate_tonneCO2perMWh is not None]
    model.eEmissionsPerScPerTp = pyo.Expression(S, T, rule=lambda m, s, t: 
                    sum(g.emission_rate_tonneCO2perMWh * m.vGEN[g, s, t] for g in G_with_emissions))
                    
    logger.info(f"   Size: {len(model.eEmissionsPerScPerTp)} expressions")
    
    logger.info(" - Expressions for generation cost per timepoint")
    model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(s.probability * g.cost_variable_USDperMWh * m.vGEN[g, s, t] for g in G for s in S)
                        )
    logger.info(f"   Size: {len(model.eGenCostPerTp)} expressions")

    model.cost_components_per_tp.append(model.eGenCostPerTp)

    model.eGenTotalCost = pyo.Expression(expr = lambda m: sum(m.eGenCostPerTp[t] * t.weight for t in T))