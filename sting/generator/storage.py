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
class Storage:
    id: int = field(default=-1, init=False)
    name: str
    technology: str
    bus: str
    cap_existing_energy_MWh: float
    cap_existing_power_MW: float
    cap_max_power_MW: float
    cost_fixed_energy_USDperkWh: float
    cost_fixed_power_USDperkW: float
    cost_variable_USDperMWh: float
    duration_hr: float
    efficiency_charge: float
    efficiency_discharge: float
    c0_USD: float
    c1_USDperMWh: float
    c2_USDperMWh2: float

    def __repr__(self):
        return f"Storage(id={self.id})"
    
    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return hash(self.id)

def construct_capacity_expansion_model(system, model, model_settings):

    N = system.bus
    T = system.tp
    S = system.sc
    E = system.ess

    # Filter energy storage units by bus
    E_AT_BUS = [[e for e in E if e.bus == n.name] for n in N]
    
    model.vDISCHA = pyo.Var(E, S, T, within=pyo.NonNegativeReals)
    model.vCHARGE = pyo.Var(E, S, T, within=pyo.NonNegativeReals)
    model.vSOC = pyo.Var(E, S, T, within=pyo.NonNegativeReals)
    model.vPCAP = pyo.Var(E, S, within=pyo.NonNegativeReals)
    model.vECAP = pyo.Var(E, S, within=pyo.NonNegativeReals)

    model.cMinEnerCapStor = pyo.Constraint(E, S, rule=lambda m, e, s: 
                                         m.vECAP[e, s] >= e.cap_existing_energy_MWh)

    model.cPowerCapStor = pyo.Constraint(E, S, rule=lambda m, e, s: 
                                         (e.cap_existing_power_MW, m.vPCAP[e, s], e.cap_max_power_MW))

    # Define constraints for energy storage systems
    model.cMaxCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vCHARGE[e, s, t] <= m.vPCAP[e, s])
    
    model.cMaxDischa = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vDISCHA[e, s, t] <= m.vPCAP[e, s])

    E_fixduration = [e for e in E if e.duration_hr > 0]

    if E_fixduration:
        model.cFixEnergyPowerRatio = pyo.Constraint(E_fixduration, S, rule=lambda m, e, s: 
                        m.vECAP[e, s] ==  e.duration_hr * m.vPCAP[e, s])
        
    model.cMaxSOC = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vSOC[e, s, t] <= m.vECAP[e, s])

    # SOC in the next time is a function of SOC in the previous time
    # with circular wrapping for the first and last timepoints within a timeseries
    model.cStateOfCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vSOC[e, s, t] == m.vSOC[e, s, T[t.prev_timepoint_id]] +
                                        t.duration_hr*(m.vCHARGE[e, s, t]*e.efficiency_charge 
                                                        - m.vDISCHA[e, s, t]*1/e.efficiency_discharge) )
    # Power generation by bus
    model.eNetDischargeAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 
                    - sum(m.vCHARGE[e, s, t] for e in E_AT_BUS[n.id]) 
                    + sum(m.vDISCHA[e, s, t] for e in E_AT_BUS[n.id]) )

    # Storage cost per timepoint
    model.eStorCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                     1/len(S)*(sum(s.probability * (sum(e.cost_variable_USDperMWh * m.vCHARGE[e, s, t] for e in E)) for s in S) ) )
    

    # Storage cost per period
    model.eStorCostPerPeriod = pyo.Expression(expr = lambda m: 
                     1/len(S)*(sum( s.probability * (sum(e.cost_fixed_power_USDperkW * m.vPCAP[e, s] * 1000 
                                                + e.cost_fixed_energy_USDperkWh * m.vECAP[e, s] * 1000 for e in E)) for s in S )) )

    # Total storage cost
    model.eStorTotalCost = pyo.Expression(expr = lambda m: 
                     sum(m.eStorCostPerTp[t] * t.weight for t in T) + m.eStorCostPerPeriod )
    