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

    logger.info(" - Decision variables of non-stochastic dispatch, i.e., dispatch with no capacity factor dependence")
    model.vGEN_NST = pyo.Var(G, T, within=pyo.NonNegativeReals)
    logger.info(f"   Size: {len(model.vGEN_NST)} variables")

    logger.info(" - Decision variables of stochastic dispatch, i.e., dispatch dependent on capacity factors")
    model.vGEN_ST = pyo.Var(G, S, T, within=pyo.NonNegativeReals)
    logger.info(f"   Size: {len(model.vGEN_ST)} variables")

    logger.info(" - Constraints on dispatch for non-variable generators (capacity factor of 1)")
    nonstochastic_gens = [g for g in G if g.site == 'no_capacity_factor' or g.site is None]
    def max_dispatch_no_cf_rule(m: pyo.ConcreteModel, g: Generator, t: Timepoint):
        return m.vGEN_NST[g, t] <= g.cap_existing_power_MW
    model.cMaxDispatchNoCf = pyo.Constraint(nonstochastic_gens, T, rule=max_dispatch_no_cf_rule)
    logger.info(f"   Size: {len(model.cMaxDispatchNoCf)} constraints")

    logger.info(" - Constraints on dispatch for variable generators")
    stochastic_gens = [g for g in G if g.site != 'no_capacity_factor' and g.site is not None]
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

        return scalefactor * m.vGEN_ST[g, s, t] <= scalefactor * capacity_factor *  g.cap_existing_power_MW

    model.cMaxDispatchCf = pyo.Constraint(stochastic_gens, S, T, rule=max_dispatch_rule)
    logger.info(f"   Size: {len(model.cMaxDispatchCf)} constraints")

    logger.info(" - Expressions for dispatch at any bus")
    stochastic_gens_at_bus = defaultdict(list)
    for g in stochastic_gens:
        stochastic_gens_at_bus[g.bus_id].append(g)
    
    nonstochastic_gens_at_bus = defaultdict(list)
    for g in nonstochastic_gens:
        nonstochastic_gens_at_bus[g.bus_id].append(g)
    model.eGenAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: sum(m.vGEN_NST[g, t] for g in nonstochastic_gens_at_bus[n.id]) 
                                                                    + sum(m.vGEN_ST[g, s, t] for g in stochastic_gens_at_bus[n.id]))
    logger.info(f"   Size: {len(model.eGenAtBus)} expressions")

    logger.info(" - Expressions for emission per timepoint per scenario expression")
    stochastic_gens_with_emissions = [g for g in stochastic_gens if g.emission_rate_tonneCO2perMWh is not None]
    nonstochastic_gens_with_emissions = [g for g in nonstochastic_gens if g.emission_rate_tonneCO2perMWh is not None]
    model.eEmissionsPerScPerTp = pyo.Expression(S, T, rule=lambda m, s, t: 
                    sum(g.emission_rate_tonneCO2perMWh * m.vGEN_ST[g, s, t] for g in stochastic_gens_with_emissions)
                    + sum(g.emission_rate_tonneCO2perMWh * m.vGEN_NST[g, t] for g in nonstochastic_gens_with_emissions))
                    
    logger.info(f"   Size: {len(model.eEmissionsPerScPerTp)} expressions")
    
    logger.info(" - Expressions for generation cost per timepoint")
    if model_settings.generator_type_costs == "quadratic":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum( (g.c2_USDperMWh2 * m.vGEN_NST[g, t]* m.vGEN_NST[g, t] + g.c1_USDperMWh * m.vGEN_NST[g, t] + g.c0_USD) for g in nonstochastic_gens)
                        +
                        sum(s.probability * (g.c2_USDperMWh2 * m.vGEN_ST[g, s, t]* m.vGEN_ST[g, s, t] + g.c1_USDperMWh * m.vGEN_ST[g, s, t] + g.c0_USD) for g in stochastic_gens for s in S) )
    elif model_settings.generator_type_costs == "linear":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(g.cost_variable_USDperMWh * m.vGEN_NST[g, t] for g in nonstochastic_gens)
                        +
                        sum(s.probability * g.cost_variable_USDperMWh * m.vGEN_ST[g, s, t] for g in stochastic_gens for s in S) )
    else:
        raise ValueError("model_settings['generator_type_costs'] must be either 'quadratic' or 'linear'.")
    logger.info(f"   Size: {len(model.eGenCostPerTp)} expressions")

    model.cost_components_per_tp.append(model.eGenCostPerTp)

    model.eGenTotalCost = pyo.Expression(expr = lambda m: sum(m.eGenCostPerTp[t] * t.weight for t in T))

@timeit    
def export_results_unit_commitment(system: System, model: pyo.ConcreteModel, output_directory: str):
    """Generator results to CSV files."""

    G: list[Generator] = system.gens.to_list()
    nonstochastic_gens = [g for g in G if g.site == 'no_capacity_factor' or g.site is None]
    stochastic_gens = [g for g in G if g.site != 'no_capacity_factor' and g.site is not None]

    # Export generator dispatch results
    (pl.DataFrame(data= [   (g.name, 
                            s.name, 
                            t.name, 
                            (pyo.value(model.vGEN_NST[g, t]) if g in nonstochastic_gens else 0) + 
                            (pyo.value(model.vGEN_ST[g, s, t]) if g in stochastic_gens else 0) ) for g in G for s in system.scenarios for t in system.timepoints],
                        schema= ['generator', 'scenario', 'timepoint', 'dispatch_MW'],
                        orient= 'row')
            .write_csv(os.path.join(output_directory, 'generator_dispatch.csv')))

    # Export emissions per scenario
    (pl.DataFrame({'scenario': [s.name for s in system.scenarios], 
                    'emissions_tonneCO2peryear': [sum(pyo.value(model.eEmissionsPerScPerTp[s, t]) * t.weight for t in system.timepoints) for s in system.scenarios]})
    .write_csv(os.path.join(output_directory, 'emissions_per_scenario.csv')))

    # Export summary of generator costs
    (pl.DataFrame({'component' : ['cost_per_timepoint_USD', 'total_cost_USD'],
                          'cost' : [  sum( pyo.value(model.eGenCostPerTp[t]) * t.weight for t in system.timepoints), 
                                            pyo.value(model.eGenTotalCost)]})
    .write_csv(os.path.join(output_directory, 'generator_costs_summary.csv')))