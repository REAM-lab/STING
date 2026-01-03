# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import pyomo.environ as pyo

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Generator:
    idx: int = field(default=-1, init=False)
    generator: str
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

    def __hash__(self):
        return hash(self.idx)

@dataclass(slots=True)
class CapacityFactor:
    idx: int = field(default=-1, init=False)
    site: str
    technology: str
    scenario: str
    timepoint: str
    capacity_factor: float

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
    """
    GV = [g for g in G if g.site != "no_capacity_factor"]

    """
    - GN is a vector of instances of generators without capacity factor profiles.
      The power generation and capacity are considered as part of 
      first-stage of the stochastic problem.
    """
    GN = [g for g in G if g.site == "no_capacity_factor"]

    G_AT_BUS = [[g for g in G if g.bus == n.bus] for n in N]
    GV_AT_BUS= [[g for g in GV if g.bus == n.bus] for n in N]
    GN_AT_BUS = [[g for g in GN if g.bus == n.bus] for n in N]

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
    
    model.cMaxGenVar = pyo.Constraint(GV, S, T, rule=lambda m, g, s, t: 
                    m.vGENV[g, s, t] <= next(cf_inst.capacity_factor for cf_inst in cf 
                                            if (cf_inst.site == g.site) and 
                                               (cf_inst.scenario == s.scenario) and 
                                               (cf_inst.timepoint == t.timepoint)
                                           ) * m.vCAPV[g, s])
    
    model.eGenAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 
                    sum(m.vGEN[g, t] for g in GN_AT_BUS[n.idx - 1]) + 
                    sum(m.vGENV[g, s, t] for g in GV_AT_BUS[n.idx - 1]) + 
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
  
    model.eCostPerPeriod += pyo.Expression(expr= lambda m: m.eGenCostPerPeriod)

    for t in T:
        model.eCostPerTp[t] = model.eCostPerTp[t] + model.eGenCostPerTp[t]
        

    model.eGenTotalCost = pyo.Expression(
                            expr = lambda m: m.eGenCostPerPeriod + sum(m.eGenCostPerTp[t] * t.weight for t in T)
                            )