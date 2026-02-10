# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar
import pyomo.environ as pyo
import polars as pl
import os
from collections import defaultdict
import logging

# -------------
# Import sting code
# --------------
from sting.utils.data_tools import pyovariable_to_df, timeit


logger = logging.getLogger(__name__)

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Generator:
    id: int = field(default=0, init=False)
    name: str
    bus: str 
    minimum_active_power_MW: float = None
    maximum_active_power_MW: float = None
    minimum_reactive_power_MVAR: float = None
    maximum_reactive_power_MVAR: float = None
    technology: str = None
    site: str = None
    cap_existing_power_MW: float = None
    cap_max_power_MW: float = None
    cost_fixed_power_USDperkW: float = None
    cost_variable_USDperMWh: float = None
    c0_USD: float = None
    c1_USDperMWh: float = None
    c2_USDperMWh2: float = None
    emission_rate_tonneCO2perMWh: float = None
    tags: ClassVar[list[str]] = ["generator"]
    bus_id: int = None
    expand_capacity: bool = None
    component_id: str = None

    def post_system_init(self, system):
        self.bus_id = next((n for n in system.bus if n.name == self.bus)).id

        if self.cap_existing_power_MW is not None and self.cap_max_power_MW is not None:
            self.expand_capacity = False if self.cap_existing_power_MW >= self.cap_max_power_MW else True

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id
    
    def __repr__(self):
        return f"Generator(id={self.id}, name='{self.name}', bus='{self.bus}')"

@dataclass(slots=True)
class CapacityFactor:
    id: int = field(default=-1, init=False)
    site: str
    scenario: str
    timepoint: str
    capacity_factor: float
    technology: str = None


def construct_ac_power_flow(acopf):
    pass

@timeit
def construct_capacity_expansion_model(system, model, model_settings):
    """Construction of generator variables, constraints, and costs."""

    S = system.sc
    T = system.tp
    G = system.gen
    cf = system.cf
    N = system.bus

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
    def max_dispatch_rule(m, g, s, t):
        if g.site != "no_capacity_factor":
                return 1e2 * m.vGEN[g, s, t] <= 1e2 * cf_lookup[(g.site, s.name, t.name)] * ( (m.vCAP[g] if g in expandable_gens else 0) + g.cap_existing_power_MW)
        else:
                return  m.vGEN[g, s, t] <= ( (m.vCAP[g] if g in expandable_gens else 0) + g.cap_existing_power_MW)

    model.cMaxDispatch = pyo.Constraint(G, S, T, rule=max_dispatch_rule)
    logger.info(f"   Size: {len(model.cMaxDispatch)} constraints")

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
def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):
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
    emissions = pl.DataFrame({'scenario': [s.name for s in system.sc], 
                            'emissions_tonneCO2peryear': [sum(pyo.value(model.eEmissionsPerScPerTp[s, t]) * t.weight for t in system.tp) for s in system.sc]})
    emissions.write_csv(os.path.join(output_directory, 'emissions_per_scenario.csv'))

    # Export summary of generator costs
    costs = pl.DataFrame({'component' : ['cost_per_timepoint_USD', 'cost_per_period_USD', 'total_cost_USD'],
                          'cost' : [  sum( pyo.value(model.eGenCostPerTp[t]) * t.weight for t in system.tp), 
                                            pyo.value(model.eGenCostPerPeriod), 
                                            pyo.value(model.eGenTotalCost)]})
    costs.write_csv(os.path.join(output_directory, 'generator_costs_summary.csv'))

def upload_built_capacities_from_csv(system, input_directory: str,  make_non_expandable: bool = True):
    """Upload built capacities from a previous capex solution. """
    
    if not os.path.exists(os.path.join(input_directory, "generator_built_capacity.csv")):
        logger.warning(f"No file named 'generator_built_capacity.csv' found in {input_directory}. Skipping upload of built capacities.")
        return
    
    generator_built_capacity = pl.read_csv(os.path.join(input_directory, "generator_built_capacity.csv"),
                                            schema_overrides={'generator': pl.String, 'built_capacity_MW': pl.Float64})
    generator_built_capacity = dict(generator_built_capacity.select("generator", "built_capacity_MW").iter_rows())

    gens_to_update = [g for g in system.gen if g.name in generator_built_capacity]

    if gens_to_update:
        for g in gens_to_update:
            g.cap_existing_power_MW += generator_built_capacity[g.name]
            if make_non_expandable:
                g.expand_capacity = False
            else:
                g.expand_capacity = False if g.cap_existing_power_MW >= g.cap_max_power_MW else True
    
    logger.info(f"> Updated existing capacities for {len(gens_to_update)} generators based on input file {os.path.join(input_directory, 'generator_built_capacity.csv')}")
    
        



