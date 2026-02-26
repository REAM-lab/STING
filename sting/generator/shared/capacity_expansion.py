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
from sting.utils.pyomo_tools import pyovariable_to_df
from sting.utils.runtime_tools import timeit
from sting.modules.capacity_expansion.utils import ModelSettings

# Set up logging
logger = logging.getLogger(__name__)
    
@timeit
def construct_capacity_expansion_model(system: System, model: pyo.ConcreteModel, model_settings: ModelSettings):
    """Construction of generator variables, constraints, and costs."""

    S: list[Scenario] = system.scenarios
    T: list[Timepoint] = system.timepoints
    G: list[Generator] = system.gens.to_list()
    cf: list[CapacityFactor] = system.capacity_factors
    N: list[Bus] = system.buses

    logger.info(" - Set of capacity-expandable generators")
    expandable_gens = [g for g in G if (g.expand_capacity and model_settings.generation_capacity_expansion)]
    logger.info(f"   Size: {len(expandable_gens)} generators")

    logger.info(" - Decision variables of capacity expansion for generators")
    model.vCAP = pyo.Var(expandable_gens, within=pyo.NonNegativeReals)
    logger.info(f"   Size: {len(model.vCAP)} variables")

    logger.info(" - Decision variables of dispatch")
    model.vGEN = pyo.Var(G, S, T, within=pyo.NonNegativeReals)
    logger.info(f"   Size: {len(model.vGEN)} variables")

    logger.info(" - Constraints on capacity expansion for generators")   
    model.cCapGenNonVar = pyo.Constraint(expandable_gens, rule=lambda m, g: m.vCAP[g] <= g.cap_max_power_MW - g.cap_existing_power_MW)
    logger.info(f"   Size: {len(model.cCapGenNonVar)} constraints")

    logger.info(" - Constraints on dispatch based on capacity factors and existing/built capacity")
    cf_lookup = {(cf_inst.site, cf_inst.scenario, cf_inst.timepoint): cf_inst.capacity_factor for cf_inst in cf}
    def max_dispatch_rule(m: pyo.ConcreteModel, g: Generator, s: Scenario, t: Timepoint):
        if g.site != "no_capacity_factor":
                if g.cap_existing_power_MW > 0 and g.cap_existing_power_MW <= 1e3 and g.cap_existing_power_MW < 10:
                    return 1e2 * m.vGEN[g, s, t] <= 1e2 * cf_lookup[(g.site, s.name, t.name)] * ( (m.vCAP[g] if g in expandable_gens else 0) + g.cap_existing_power_MW)
                else:
                    return m.vGEN[g, s, t] <= cf_lookup[(g.site, s.name, t.name)] * ( (m.vCAP[g] if g in expandable_gens else 0) + g.cap_existing_power_MW)
        else:
                if g.cap_existing_power_MW > 100:
                    return 1e-2 * m.vGEN[g, s, t] <= ( 1e-2 * (m.vCAP[g] if g in expandable_gens else 0) + g.cap_existing_power_MW)
                else:
                    return m.vGEN[g, s, t] <= ( (m.vCAP[g] if g in expandable_gens else 0) + g.cap_existing_power_MW)

    model.cMaxDispatch = pyo.Constraint(G, S, T, rule=max_dispatch_rule)
    logger.info(f"   Size: {len(model.cMaxDispatch)} constraints")

    logger.info(" - Constraints on forced dispatch requirements specified for certain generators")
    forced_dispatch_gens = [g for g in G if g.forced_dispatch_MW is not None]
    model.cForcedDispatch = pyo.Constraint(forced_dispatch_gens, S, T, rule=lambda m, g, s, t: m.vGEN[g, s, t] == g.forced_dispatch_MW)
    logger.info(f"   Size: {len(model.cForcedDispatch)} constraints")

    logger.info(" - Expressions for dispatch at any bus")
    gens_at_bus = defaultdict(list)
    for g in G:
        gens_at_bus[g.bus_id].append(g)
    model.eGenAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: sum(m.vGEN[g, s, t] for g in gens_at_bus[n.id]))
    logger.info(f"   Size: {len(model.eGenAtBus)} expressions")

    logger.info(" - Expressions for emission per timepoint per scenario expression")
    gens_with_emissions = [g for g in G if g.emission_rate_tonneCO2perMWh is not None]
    model.eEmissionsPerScPerTp = pyo.Expression(S, T, rule=lambda m, s, t: 
                    sum(g.emission_rate_tonneCO2perMWh * m.vGEN[g, s, t] for g in gens_with_emissions))
    logger.info(f"   Size: {len(model.eEmissionsPerScPerTp)} expressions")

    logger.info(" - Expressions for generation cost per timepoint")
    if model_settings.generator_type_costs == "quadratic":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(s.probability * (g.c2_USDperMWh2 * m.vGEN[g, s, t]* m.vGEN[g, s, t] + g.c1_USDperMWh * m.vGEN[g, s, t] + g.c0_USD) for g in G for s in S) )
    elif model_settings.generator_type_costs == "linear":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(s.probability * g.cost_variable_USDperMWh * m.vGEN[g, s, t] for g in G for s in S) )
    else:
        raise ValueError("model_settings['generator_type_costs'] must be either 'quadratic' or 'linear'.")
    logger.info(f"   Size: {len(model.eGenCostPerTp)} expressions")

    model.cost_components_per_tp.append(model.eGenCostPerTp)
    
    logger.info(" - Expressions for generation cost per period")
    # The factor of 1000 is to convert from kW to MW, since the cost_fixed_power is per kW and the capacity variables are in MW.
    model.eGenCostPerPeriod = pyo.Expression(expr = lambda m: sum(g.cost_fixed_power_USDperkW * m.vCAP[g] * 1000 for g in expandable_gens) )
    model.cost_components_per_period.append(model.eGenCostPerPeriod)

    model.eGenTotalCost = pyo.Expression(expr = lambda m: m.eGenCostPerPeriod + sum(m.eGenCostPerTp[t] * t.weight for t in T))
    

@timeit    
def export_results_capacity_expansion(system: System, model: pyo.ConcreteModel, output_directory: str):
    """Generator results to CSV files."""

    # Export generator dispatch results
    pyovariable_to_df(model.vGEN, 
                      dfcol_to_field={'generator': 'name', 'scenario': 'name', 'timepoint': 'name'}, 
                      value_name='dispatch_MW', 
                      csv_filepath=os.path.join(output_directory, 'generator_dispatch.csv'))

    # Export generator capacity results
    pyovariable_to_df(model.vCAP, 
                      dfcol_to_field={'generator': 'name'}, 
                      value_name='built_capacity_MW', 
                      csv_filepath=os.path.join(output_directory, 'generator_built_capacity.csv'))
    
    # Export emissions per scenario
    emissions = pl.DataFrame({'scenario': [s.name for s in system.scenarios], 
                            'emissions_tonneCO2peryear': [sum(pyo.value(model.eEmissionsPerScPerTp[s, t]) * t.weight for t in system.timepoints) for s in system.scenarios]})
    emissions.write_csv(os.path.join(output_directory, 'emissions_per_scenario.csv'))

    # Export summary of generator costs
    costs = pl.DataFrame({'component' : ['cost_per_timepoint_USD', 'cost_per_period_USD', 'total_cost_USD'],
                          'cost' : [  sum( pyo.value(model.eGenCostPerTp[t]) * t.weight for t in system.timepoints), 
                                            pyo.value(model.eGenCostPerPeriod), 
                                            pyo.value(model.eGenTotalCost)]})
    costs.write_csv(os.path.join(output_directory, 'generator_costs_summary.csv'))

def upload_built_capacities_from_csv(system: System, input_directory: str,  make_non_expandable: bool = True):
    """Upload built capacities from a previous capex solution. """
    
    if not os.path.exists(os.path.join(input_directory, "generator_built_capacity.csv")):
        logger.warning(f"No file named 'generator_built_capacity.csv' found in {input_directory}. Skipping upload of built capacities.")
        return
    
    generator_built_capacity = pl.read_csv(os.path.join(input_directory, "generator_built_capacity.csv"),
                                            schema_overrides={'generator': pl.String, 'built_capacity_MW': pl.Float64})
    generator_built_capacity = dict(generator_built_capacity.select("generator", "built_capacity_MW").iter_rows())

    G: list[Generator] = system.gens.to_list()
    gens_to_update = [g for g in G if g.name in generator_built_capacity]

    if gens_to_update:
        for g in gens_to_update:
            g.cap_existing_power_MW += generator_built_capacity[g.name]
            if make_non_expandable:
                g.expand_capacity = False
            else:
                g.expand_capacity = False if g.cap_existing_power_MW >= g.cap_max_power_MW else True
    
    logger.info(f"> Updated existing capacities for {len(gens_to_update)} generators based on input file {os.path.join(input_directory, 'generator_built_capacity.csv')}")