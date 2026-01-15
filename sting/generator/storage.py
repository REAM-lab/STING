# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import pyomo.environ as pyo
import os
import polars as pl
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
    expand_capacity: bool = True

    def post_system_init(self, system):
        self.expand_capacity = False if self.cap_existing_power_MW >= self.cap_max_power_MW else True

    def __repr__(self):
        return f"Storage(id={self.id})"
    
    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id

@timeit
def construct_capacity_expansion_model(system, model, model_settings):
    """Construction of storage variables, constraints, and costs."""

    N = system.bus
    T = system.tp
    S = system.sc
    E = system.ess

    logger.info(" - Decision variables")
    if model_settings["consider_single_storage_injection"]:
            model.vDISCHA = pyo.Var(E, S, T, within=pyo.Reals)            
    else:
            model.vDISCHA = pyo.Var(E, S, T, within=pyo.NonNegativeReals)
            model.vCHARGE = pyo.Var(E, S, T, within=pyo.NonNegativeReals)

    model.vSOC = pyo.Var(E, S, T, within=pyo.NonNegativeReals)
    model.vPCAP = pyo.Var(E, S, within=pyo.NonNegativeReals)
    model.vECAP = pyo.Var(E, S, within=pyo.NonNegativeReals)
    logger.info(f"""   Size: {len(model.vDISCHA) + (len(model.vCHARGE) if hasattr(model, 'vCHARGE') else 0) 
                            + len(model.vSOC) + len(model.vPCAP) + len(model.vECAP)} variables""")

    for ess in E:
         if (not ess.expand_capacity):
            model.vPCAP[ess, :].fix(0.0)
            model.vECAP[ess, :].fix(0.0)

    logger.info(" - Energy capacity constraints")
    def cEnerCapStor_rule(m, e, s):
        if e.expand_capacity:
            return  m.vECAP[e, s] <= e.cap_max_energy_MWh - e.cap_existing_energy_MWh
        else:
            return pyo.Constraint.Skip
        
    model.cEnerCapStor = pyo.Constraint(E, S, rule=cEnerCapStor_rule)
    logger.info(f"   Size: {len(model.cEnerCapStor)} constraints")

    logger.info(" - Power capacity constraints")
    def cPowerCapStor_rule(m, e, s):
        if e.expand_capacity:
            return  m.vPCAP[e, s] <= e.cap_max_power_MW - e.cap_existing_power_MW
        else:
            return pyo.Constraint.Skip
        
    model.cPowerCapStor = pyo.Constraint(E, S, rule=cPowerCapStor_rule)
    logger.info(f"   Size: {len(model.cPowerCapStor)} constraints")

    logger.info(" - Maximum charge and discharge constraints")
    if model_settings["consider_single_storage_injection"]:
        model.cMaxCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                                          m.vDISCHA[e, s, t] >= - (m.vPCAP[e, s] + e.cap_existing_power_MW))
        model.cMaxDischa = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                                          m.vDISCHA[e, s, t] <= m.vPCAP[e, s] + e.cap_existing_power_MW)
    else:
        model.cMaxCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vCHARGE[e, s, t] <= m.vPCAP[e, s] + e.cap_existing_power_MW)
    
        model.cMaxDischa = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vDISCHA[e, s, t] <= m.vPCAP[e, s] + e.cap_existing_power_MW)
    logger.info(f"   Size: {len(model.cMaxCharge) + len(model.cMaxDischa)} constraints")

    E_fixduration_expandable = [e for e in E if ((e.duration_hr > 0) and (e.expand_capacity))]
    if E_fixduration_expandable:
        logger.info(" - Energy-power ratio constraints for expandable storage with fixed duration")
        model.cFixEnergyPowerRatio = pyo.Constraint(E_fixduration_expandable, S, rule=lambda m, e, s: 
                        (m.vECAP[e, s] + e.cap_existing_energy_MWh) ==  e.duration_hr * (m.vPCAP[e, s] + e.cap_existing_power_MW))
        logger.info(f"   Size: {len(model.cFixEnergyPowerRatio)} constraints")
    
    logger.info(" - Maximum state of charge constraints")
    model.cMaxSOC = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vSOC[e, s, t] <= (m.vECAP[e, s] + e.cap_existing_energy_MWh))
    logger.info(f"   Size: {len(model.cMaxSOC)} constraints")

    # SOC in the next time is a function of SOC in the previous time
    # with circular wrapping for the first and last timepoints within a timeseries
    logger.info(" - State of charge constraints")
    E_AT_BUS = [[e for e in E if e.bus == n.name] for n in N]
    if model_settings["consider_single_storage_injection"]:
        model.cStateOfCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vSOC[e, s, t] == m.vSOC[e, s, T[t.prev_timepoint_id]] +
                                        t.duration_hr*(- m.vDISCHA[e, s, t]) )
        model.eNetDischargeAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 
                    + sum(m.vDISCHA[e, s, t] for e in E_AT_BUS[n.id]) )
        
    else:
        model.cStateOfCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vSOC[e, s, t] == m.vSOC[e, s, T[t.prev_timepoint_id]] +
                                        t.duration_hr*(m.vCHARGE[e, s, t]*e.efficiency_charge 
                                                        - m.vDISCHA[e, s, t]*1/e.efficiency_discharge) )
        model.eNetDischargeAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 
                    - sum(m.vCHARGE[e, s, t] for e in E_AT_BUS[n.id]) 
                    + sum(m.vDISCHA[e, s, t] for e in E_AT_BUS[n.id]) )
    logger.info(f"   Size: {len(model.cStateOfCharge) + len(model.eNetDischargeAtBus)} constraints")

    logger.info(" - Storage cost per timepoint expressions")
    if model_settings["consider_single_storage_injection"]:
        model.eStorCostPerTp = pyo.Expression(T, rule=lambda m, t: 0.0 )
    else:
        model.eStorCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                1/len(S)*(sum(s.probability * (sum(e.cost_variable_USDperMWh * (m.vCHARGE[e, s, t] + m.vDISCHA[e, s, t]) for e in E)) for s in S) ) )
    
    model.cost_components_per_tp.append(model.eStorCostPerTp)
    logger.info(f"   Size: {len(model.eStorCostPerTp)} expressions")

    logger.info(" - Storage cost per period expression")
    model.eStorCostPerPeriod = pyo.Expression(expr = lambda m: 
                     1/len(S)*(sum( s.probability * (sum(e.cost_fixed_power_USDperkW * m.vPCAP[e, s] * 1000 
                                                + e.cost_fixed_energy_USDperkWh * m.vECAP[e, s] * 1000 for e in E)) for s in S )) )
    
    model.cost_components_per_period.append(model.eStorCostPerPeriod)
    logger.info(f"   Size: {len(model.eStorCostPerPeriod)} expressions")

    # Total storage cost
    model.eStorTotalCost = pyo.Expression(expr = lambda m: 
                     sum(m.eStorCostPerTp[t] * t.weight for t in T) + m.eStorCostPerPeriod )

@timeit
def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):
    """Storage results to CSV files."""

    # Export discharge and charge results
    df1 = pyovariable_to_df(model.vDISCHA, 
                            dfcol_to_field={'storage': 'name', 'scenario': 'name', 'timepoint': 'name'}, 
                            value_name='discharge_MW')
    
    if hasattr(model, 'vCHARGE'):
        df2 = pyovariable_to_df(model.vCHARGE, 
                            dfcol_to_field={'storage': 'name', 'scenario': 'name', 'timepoint': 'name'}, 
                            value_name='charge_MW')
    
        df1 = df1.join(df2, on=['storage', 'scenario', 'timepoint'])
    
    # Join with state of charge results
    df3 = pyovariable_to_df(model.vSOC, 
                            dfcol_to_field={'storage': 'name', 'scenario': 'name', 'timepoint': 'name'}, 
                            value_name='state_of_charge_MWh')
    df1 = df1.join(df3, on=['storage', 'scenario', 'timepoint'])

    df1.write_csv(os.path.join(output_directory, 'storage_dispatch.csv'))

    # Export storage capacity results
    df1 = pyovariable_to_df(model.vPCAP, 
                            dfcol_to_field={'storage': 'name', 'scenario': 'name'}, 
                            value_name='power_capacity_MW')
    
    df2 = pyovariable_to_df(model.vECAP, 
                            dfcol_to_field={'storage': 'name', 'scenario': 'name'}, 
                            value_name='energy_capacity_MWh')
    
    df = df1.join(df2, on=['storage', 'scenario'])
    df.write_csv(os.path.join(output_directory, 'storage_capacity.csv'))

    # Export summary of storage costs
    costs = pl.DataFrame({'component' : ['CostPerTimepoint_USD', 'CostPerPeriod_USD', 'TotalCost_USD'],
                          'cost' : [  sum( pyo.value(model.eStorCostPerTp[t]) * t.weight for t in system.tp), 
                                            pyo.value(model.eStorCostPerPeriod), 
                                            pyo.value(model.eStorTotalCost)]})
    costs.write_csv(os.path.join(output_directory, 'storage_costs_summary.csv'))
