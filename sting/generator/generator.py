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
    technology: str
    bus: str
    site: str
    cap_existing_power_MW: float
    cap_max_power_MW: float
    cost_fixed_power_USDperkW: float
    cost_variable_USDperMWh: float
    c0_USD: float
    c1_USDperMWh: float
    c2_USDperMWh2: float
    tags: ClassVar[list[str]] = ["generator"]
    bus_id: int = None
    expand_capacity: bool = True

    def post_system_init(self, system):
        self.bus_id = next((n for n in system.bus if n.name == self.bus)).id
        self.expand_capacity = False if self.cap_existing_power_MW >= self.cap_max_power_MW else True

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id

@dataclass(slots=True)
class CapacityFactor:
    id: int = field(default=-1, init=False)
    site: str
    scenario: str
    timepoint: str
    capacity_factor: float
    technology: str = None

@timeit
def construct_capacity_expansion_model(system, model, model_settings):
    """Construction of generator variables, constraints, and costs."""

    S = system.sc
    T = system.tp
    G = system.gen
    cf = system.cf
    N = system.bus

    logger.info(" - Sets of generators for first-stage and second-stage decisions")
    """
    - GV is a vector of instances of generators with capacity factor profiles.
        Power generation and capacity of these generators are considered random variables 
        in the second-stage of the stochastic problem.
    - GN is a vector of instances of generators without capacity factor profiles.
      The power generation and capacity are considered as part of 
      first-stage of the stochastic problem.
    """
    GV = []
    GN = []
    for g in G:
        if g.site == "no_capacity_factor":
            GN.append(g)
        else:
            GV.append(g)

    GV_at_bus = defaultdict(list)
    GN_at_bus = defaultdict(list)

    for g in GV:
        GV_at_bus[g.bus_id].append(g)
    for g in GN:
        GN_at_bus[g.bus_id].append(g)

    logger.info(" - Decision variables of dispatch and capacity")
    model.vGEN = pyo.Var(GN, T, within=pyo.NonNegativeReals)
    model.vCAP = pyo.Var(GN, within=pyo.NonNegativeReals)
    model.vGENV = pyo.Var(GV, S, T, within=pyo.NonNegativeReals)
    model.vCAPV = pyo.Var(GV, S, within=pyo.NonNegativeReals)
    logger.info(f"   Size: {len(model.vGEN) + len(model.vCAP) + len(model.vGENV) + len(model.vCAPV)} variables")

    for g in GN:
         if (not g.expand_capacity):
            model.vCAP[g].fix(0.0)
    
    for g in GV:
         if (not g.expand_capacity):
            model.vCAPV[g, :].fix(0.0)

    logger.info(" - Maximum built capacity constraints")    
    def cCapGenVar_rule(m, g, s):
        if g.expand_capacity:
            return  m.vCAPV[g, s] <= g.cap_max_power_MW - g.cap_existing_power_MW
        else:
            return pyo.Constraint.Skip

    model.cCapGenVar = pyo.Constraint(GV, S, rule=cCapGenVar_rule)

    def cCapGenNonVar_rule(m, g):
        if g.expand_capacity:
            return  m.vCAP[g] <= g.cap_max_power_MW - g.cap_existing_power_MW
        else:
            return pyo.Constraint.Skip
    
    model.cCapGenNonVar = pyo.Constraint(GN, rule=cCapGenNonVar_rule)
    logger.info(f"   Size: {len(model.cCapGenNonVar) + len(model.cCapGenVar)} constraints")

    logger.info(" - Maximum dispatch constraints")    
    model.cMaxGenNonVar = pyo.Constraint(GN, T, rule=lambda m, g, t: 
                    m.vGEN[g, t] <= m.vCAP[g] + g.cap_existing_power_MW)
    logger.info(f"   Size: {len(model.cMaxGenNonVar)} constraints")
    
    logger.info(" - Capacity factor constraints")
    cf_lookup = {(cf_inst.site, cf_inst.scenario, cf_inst.timepoint): cf_inst.capacity_factor for cf_inst in cf}
    model.cMaxGenVar = pyo.Constraint(GV, S, T, rule=lambda m, g, s, t: 
                    m.vGENV[g, s, t] <= cf_lookup[(g.site, s.name, t.name)] * (m.vCAPV[g, s] + g.cap_existing_power_MW))
    logger.info(f"   Size: {len(model.cMaxGenVar)} constraints")
    
    logger.info(" - Dispatch at bus expressions")
    model.eGenAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 
                    sum(m.vGEN[g, t] for g in GN_at_bus[n.id]) + 
                    sum(m.vGENV[g, s, t] for g in GV_at_bus[n.id]))
                
    logger.info(f"   Size: {len(model.eGenAtBus)} expressions")

    logger.info(" - Generation cost per timepoint expressions")
    if model_settings["gen_costs"] == "quadratic":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(g.c2_USDperMWh2 * m.vGEN[g, t]* m.vGEN[g, t] + g.c1_USDperMWh * m.vGEN[g, t] + g.c0_USD for g in GN) + 
                        1/len(S) * sum(s.probability * (g.c2_USDperMWh2 * m.vGENV[g, s, t]* m.vGENV[g, s, t] + g.c1_USDperMWh * m.vGENV[g, s, t] + g.c0_USD ) for g in GV for s in S))
    elif model_settings["gen_costs"] == "linear":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(g.cost_variable_USDperMWh * m.vGEN[g, t] for g in GN) + 
                        1/len(S) * sum(s.probability * g.cost_variable_USDperMWh * m.vGENV[g, s, t] for g in GV for s in S) )
    else:
        raise ValueError("model_settings['gen_costs'] must be either 'quadratic' or 'linear'.")
    logger.info(f"   Size: {len(model.eGenCostPerTp)} expressions")

    model.cost_components_per_tp.append(model.eGenCostPerTp)
    
    logger.info(" - Generation cost per period expression")
    model.eGenCostPerPeriod = pyo.Expression(
                                expr = lambda m: sum(g.cost_fixed_power_USDperkW * m.vCAP[g] * 1000 for g in GN) + 
                                       1/len(S) * sum( (s.probability * g.cost_fixed_power_USDperkW * m.vCAPV[g, s] * 1000) for g in GV for s in S )
                                )
    model.cost_components_per_period.append(model.eGenCostPerPeriod)

    model.eGenTotalCost = pyo.Expression(
                            expr = lambda m: m.eGenCostPerPeriod + sum(m.eGenCostPerTp[t] * t.weight for t in T)
                            )
    

        
@timeit    
def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):
    """Generator results to CSV files."""

    # Export generator dispatch results
    pyovariable_to_df(model.vGEN, 
                      dfcol_to_field={'generator': 'name', 'timepoint': 'name'}, 
                      value_name='Dispatch_MW', 
                      csv_filepath=os.path.join(output_directory, 'generator_dispatch.csv'))

    pyovariable_to_df(model.vGENV, 
                      dfcol_to_field={'generator': 'name', 'scenario': 'name', 'timepoint': 'name'}, 
                      value_name='Dispatch_MW', 
                      csv_filepath=os.path.join(output_directory, 'variable_generator_dispatch.csv'))

    # Export generator capacity results
    pyovariable_to_df(model.vCAP, 
                      dfcol_to_field={'generator': 'name'}, 
                      value_name='Capacity_MW', 
                      csv_filepath=os.path.join(output_directory, 'generator_capacity.csv'))

    pyovariable_to_df(model.vCAPV, 
                      dfcol_to_field={'generator': 'name', 'scenario': 'name'}, 
                      value_name='Capacity_MW', 
                      csv_filepath=os.path.join(output_directory, 'variable_generator_capacity.csv'))

    # Export summary of generator costs
    costs = pl.DataFrame({'component' : ['CostPerTimepoint_USD', 'CostPerPeriod_USD', 'TotalCost_USD'],
                          'cost' : [  sum( pyo.value(model.eGenCostPerTp[t]) * t.weight for t in system.tp), 
                                            pyo.value(model.eGenCostPerPeriod), 
                                            pyo.value(model.eGenTotalCost)]})
    costs.write_csv(os.path.join(output_directory, 'generator_costs_summary.csv'))





