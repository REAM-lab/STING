# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar
import pyomo.environ as pyo
import polars as pl
import os
from collections import defaultdict

# -------------
# Import sting code
# --------------
from sting.utils.data_tools import pyovariable_to_df

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

    def assign_indices(self, system):
        self.bus_id = next((n for n in system.bus if n.name == self.bus)).id

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

def construct_capacity_expansion_model(system, model, model_settings):

    S = system.sc
    T = system.tp
    G = system.gen
    cf = system.cf
    N = system.bus

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

    model.vGEN = pyo.Var(GN, T, within=pyo.NonNegativeReals)
    model.vCAP = pyo.Var(GN, within=pyo.NonNegativeReals)
    model.vGENV = pyo.Var(GV, S, T, within=pyo.NonNegativeReals)
    model.vCAPV = pyo.Var(GV, S, within=pyo.NonNegativeReals)

    if model_settings["consider_shedding"]:
        model.vSHED = pyo.Var(N, S, T, within=pyo.NonNegativeReals)
    
    model.cCapGenVar = pyo.Constraint(GV, S, rule= lambda m, g, s: 
                    (g.cap_existing_power_MW, m.vCAPV[g, s], g.cap_max_power_MW))
    
    model.cCapGenNonVar = pyo.Constraint(GN, rule= lambda m, g: 
                    (g.cap_existing_power_MW, m.vCAP[g], g.cap_max_power_MW))
    
    model.cMaxGenNonVar = pyo.Constraint(GN, T, rule=lambda m, g, t: 
                    m.vGEN[g, t] <= m.vCAP[g])
    
    cf_lookup = {(cf_inst.site, cf_inst.scenario, cf_inst.timepoint): cf_inst.capacity_factor for cf_inst in cf}
    model.cMaxGenVar = pyo.Constraint(GV, S, T, rule=lambda m, g, s, t: 
                    m.vGENV[g, s, t] <= cf_lookup[(g.site, s.name, t.name)] * m.vCAPV[g, s])
    
    model.eGenAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 
                    sum(m.vGEN[g, t] for g in GN_at_bus[n.id]) + 
                    sum(m.vGENV[g, s, t] for g in GV_at_bus[n.id]) + 
                    (m.vSHED[n, s, t] if model_settings["consider_shedding"] else 0)
                )

    # The weighted operational costs of running each generator
    if model_settings["gen_costs"] == "quadratic":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(g.c2_USDperMWh2 * m.vGEN[g, t]* m.vGEN[g, t] + g.c1_USDperMWh * m.vGEN[g, t] + g.c0_USD for g in GN) + 
                        1/len(S) * sum(s.probability * (g.c2_USDperMWh2 * m.vGENV[g, s, t]* m.vGENV[g, s, t] + g.c1_USDperMWh * m.vGENV[g, s, t] + g.c0_USD ) for g in GV for s in S))
    elif model_settings["gen_costs"] == "linear":
        model.eGenCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                        sum(g.cost_variable_USDperMWh * m.vGEN[g, t] for g in GN) + 
                        1/len(S) * sum(s.probability * g.cost_variable_USDperMWh * m.vGENV[g, s, t] for g in GV for s in S) + 
                        (sum(5000 * m.vSHED[n, s, t] for n in N for s in S) if model_settings["consider_shedding"] else 0) )
    else:
        raise ValueError("model_settings['gen_costs'] must be either 'quadratic' or 'linear'.")
    
    # Fixed costs 
    model.eGenCostPerPeriod = pyo.Expression(
                                expr = lambda m: sum(g.cost_fixed_power_USDperkW * m.vCAP[g] * 1000 for g in GN) + 
                                       1/len(S) * sum( (s.probability * g.cost_fixed_power_USDperkW * m.vCAPV[g, s] * 1000) for g in GV for s in S )
                                )

    model.eGenTotalCost = pyo.Expression(
                            expr = lambda m: m.eGenCostPerPeriod + sum(m.eGenCostPerTp[t] * t.weight for t in T)
                            )
    
def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):

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

    # Export load shedding results if it is existing
    if hasattr(model, 'vSHED'):
        pyovariable_to_df(model.vSHED, 
                          dfcol_to_field={'bus': 'name', 'scenario': 'name', 'timepoint': 'name'}, 
                          value_name='Load_Shed_MW', 
                          csv_filepath=os.path.join(output_directory, 'load_shedding.csv'))

    
    # Export summary of generator costs
    costs = pl.DataFrame({'component' : ['CostPerTimepoint_USD', 'CostPerPeriod_USD', 'TotalCost_USD'],
                          'cost' : [  sum( pyo.value(model.eGenCostPerTp[t]) * t.weight for t in system.tp), 
                                            pyo.value(model.eGenCostPerPeriod), 
                                            pyo.value(model.eGenTotalCost)]})
    costs.write_csv(os.path.join(output_directory, 'generator_costs_summary.csv'))


