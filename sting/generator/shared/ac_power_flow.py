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
from sting.generator.core import Generator
from sting.timescales.core import Timepoint
from sting.system.core import System
from sting.utils.runtime_tools import timeit
from sting.modules.power_flow.utils import ModelSettings

# Set up logging
logger = logging.getLogger(__name__)

def construct_ac_power_flow_model(system: System, model: pyo.ConcreteModel, model_settings: ModelSettings):

    T: list[Timepoint] = system.timepoints
    G: list[Generator] = system.gens.to_list()
    N: list[Bus] = system.buses

    logger.info(" - Decision variables of active power and reactive power for generators")
    model.vPG = pyo.Var(G, T, 
                           within=pyo.Reals, 
                           bounds=lambda m, g, t: (0, 10) )
    model.vQG = pyo.Var(G, T,
                            within=pyo.Reals, 
                            bounds=lambda m, g, t: (-10, 10) )
    logger.info(f"   Size: {len(model.vPG) + len(model.vQG)} variables")

    logger.info(" - Expressions for dispatch of active power at any bus")
    gens_at_bus = defaultdict(list)
    for g in G:
        gens_at_bus[g.bus_id].append(g)
    model.eActiveDispatchAtBus = pyo.Expression(N, T, rule=lambda m, n, t: sum(m.vPG[g, t] for g in gens_at_bus[n.id]))
    logger.info(f"   Size: {len(model.eActiveDispatchAtBus)} expressions")

    logger.info(" - Expressions for dispatch of reactive power at any bus")
    model.eReactiveDispatchAtBus = pyo.Expression(N, T, rule=lambda m, n, t: sum(m.vQG[g, t] for g in gens_at_bus[n.id]))
    logger.info(f"   Size: {len(model.eReactiveDispatchAtBus)} expressions")

    logger.info(" - Expressions for generation cost per timepoint")
    if model_settings.generator_type_costs == "quadratic":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(g.c2_USDperMWh2 * m.vPG[g, t]* m.vPG[g, t] + g.c1_USDperMWh * m.vPG[g, t] + g.c0_USD for g in G) )
    elif model_settings.generator_type_costs == "linear":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(g.cost_variable_USDperMWh * m.vPG[g, t] for g in G) )
    else:
        raise ValueError("model_settings['generator_type_costs'] must be either 'quadratic' or 'linear'.")
    logger.info(f"   Size: {len(model.eGenCostPerTp)} expressions")

    model.cost_components_per_tp.append(model.eGenCostPerTp)

@timeit
def export_results_ac_power_flow(system: System, model: pyo.ConcreteModel, output_directory: str):
    """Export generator dispatch results to CSV files."""

    G: list[Generator] = system.gens.to_list() 
    T: list[Timepoint] = system.timepoints

    # Export generator dispatch results
    df = pl.DataFrame(data = [ (g.id, g.type_, g.name, t.name, pyo.value(model.vPG[g, t]), pyo.value(model.vQG[g, t])) for g in G for t in T],
                        schema = ['id', 'type', 'generator', 'timepoint', 'active_power_MW', 'reactive_power_MVAR'],
                        orient = 'row')
    df.write_csv(os.path.join(output_directory, 'generator_dispatch.csv'))
