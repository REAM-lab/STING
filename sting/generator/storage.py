# -------------
# Import python packages
# --------------
from collections import defaultdict
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
    bus_id: int = None

    def post_system_init(self, system):
        self.expand_capacity = False if self.cap_existing_power_MW >= self.cap_max_power_MW else True
        self.bus_id = next((n for n in system.bus if n.name == self.bus)).id

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

    if model_settings.single_storage_injection:
            logger.info(" - Decision variables of discharge (single injection)")
            model.vDISCHA = pyo.Var(E, S, T, within=pyo.Reals)            
            logger.info(f"   Size: {len(model.vDISCHA)} variables")
    else:
            logger.info(" - Decision variables of discharge and charge")
            model.vDISCHA = pyo.Var(E, S, T, within=pyo.NonNegativeReals)
            model.vCHARGE = pyo.Var(E, S, T, within=pyo.NonNegativeReals)
            logger.info(f"   Size: {len(model.vDISCHA) + len(model.vCHARGE)} variables")

    logger.info(" - Decision variables of state of charge")
    model.vSOC = pyo.Var(E, S, T, within=pyo.NonNegativeReals)
    logger.info(f"   Size: {len(model.vSOC)} variables")

    logger.info(" - Set of capacity-expandable storage units")
    expandable_ess = [e for e in E if (e.expand_capacity and model_settings.storage_capacity_expansion)]
    logger.info(f"   Size: {len(expandable_ess)} storage units")

    logger.info(" - Decision variables of power and energy capacity expansion")
    model.vPCAP = pyo.Var(expandable_ess, within=pyo.NonNegativeReals)
    model.vECAP = pyo.Var(expandable_ess, within=pyo.NonNegativeReals)
    logger.info(f"   Size: {len(model.vPCAP) + len(model.vECAP)} variables")

    logger.info(" - Constraints on power capacity expansion")        
    model.cPowerCapStor = pyo.Constraint(expandable_ess, rule=lambda m, e: m.vPCAP[e] <= e.cap_max_power_MW - e.cap_existing_power_MW)
    logger.info(f"   Size: {len(model.cPowerCapStor)} constraints")

    logger.info(" - Constraints on storage dispatch")
    if model_settings.single_storage_injection:
        model.cMaxCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                                          m.vDISCHA[e, s, t] >= - ( (m.vPCAP[e] if e in expandable_ess else 0) + e.cap_existing_power_MW))
        model.cMaxDischarge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                                          m.vDISCHA[e, s, t] <= ( (m.vPCAP[e] if e in expandable_ess else 0) + e.cap_existing_power_MW))
    else:
        model.cMaxCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vCHARGE[e, s, t] <= ( (m.vPCAP[e] if e in expandable_ess else 0) + e.cap_existing_power_MW))
    
        model.cMaxDischarge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vDISCHA[e, s, t] <= ( (m.vPCAP[e] if e in expandable_ess else 0) + e.cap_existing_power_MW))
    logger.info(f"   Size: {len(model.cMaxCharge) + len(model.cMaxDischarge)} constraints")

    E_fixduration_expandable = [e for e in expandable_ess if (e.duration_hr > 0)]
    if E_fixduration_expandable:
        logger.info(" - Energy-power ratio constraints for expandable storage with fixed duration")
        model.cFixEnergyPowerRatio = pyo.Constraint(E_fixduration_expandable, rule=lambda m, e: 
                        (m.vECAP[e] + e.cap_existing_energy_MWh) ==  e.duration_hr * (m.vPCAP[e] + e.cap_existing_power_MW))
        logger.info(f"   Size: {len(model.cFixEnergyPowerRatio)} constraints")
    
    logger.info(" - Constraints on maximum state of charge")
    model.cMaxSOC = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vSOC[e, s, t] <= (m.vECAP[e] if e in expandable_ess else 0) + e.cap_existing_energy_MWh)
    logger.info(f"   Size: {len(model.cMaxSOC)} constraints")

    # SOC in the next time is a function of SOC in the previous time
    # with circular wrapping for the first and last timepoints within a timeseries
    logger.info(" - Constraints on evolution of state of charge across timepoints")
    ess_at_bus = defaultdict(list)
    for e in E:
        ess_at_bus[e.bus_id].append(e)

    if model_settings.single_storage_injection:
        model.cStateOfCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vSOC[e, s, t] == m.vSOC[e, s, T[t.prev_timepoint_id]] + t.duration_hr * (- m.vDISCHA[e, s, t]) )
        model.eNetDischargeAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: sum(m.vDISCHA[e, s, t] for e in ess_at_bus[n.id]) )
    else:
        model.cStateOfCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vSOC[e, s, t] == m.vSOC[e, s, T[t.prev_timepoint_id]] +
                                        t.duration_hr * (m.vCHARGE[e, s, t]*e.efficiency_charge 
                                                        - m.vDISCHA[e, s, t]*1/e.efficiency_discharge) )
        model.eNetDischargeAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 
                    - sum(m.vCHARGE[e, s, t] for e in ess_at_bus[n.id]) 
                    + sum(m.vDISCHA[e, s, t] for e in ess_at_bus[n.id]) )
    logger.info(f"   Size: {len(model.cStateOfCharge) + len(model.eNetDischargeAtBus)} constraints")

    logger.info(" - Expressions for storage cost per timepoint")
    if model_settings.single_storage_injection:
        model.eStorCostPerTp = pyo.Expression(T, rule=lambda m, t: 0.0 )
    else:
        model.eStorCostPerTp = pyo.Expression(T, rule=lambda m, t: 
                (sum(s.probability * (sum(e.cost_variable_USDperMWh * (m.vCHARGE[e, s, t] + m.vDISCHA[e, s, t]) for e in E)) for s in S) ) )
    
    model.cost_components_per_tp.append(model.eStorCostPerTp)
    logger.info(f"   Size: {len(model.eStorCostPerTp)} expressions")

    logger.info(" - Expression for storage cost per period")
    model.eStorCostPerPeriod = pyo.Expression(expr = lambda m: 
                     sum(e.cost_fixed_power_USDperkW * m.vPCAP[e] * 1000 + e.cost_fixed_energy_USDperkWh * m.vECAP[e] * 1000 for e in expandable_ess) )
    
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
                            dfcol_to_field={'storage': 'name'}, 
                            value_name='built_power_capacity_MW')
    
    df2 = pyovariable_to_df(model.vECAP, 
                            dfcol_to_field={'storage': 'name'}, 
                            value_name='built_energy_capacity_MWh')
    
    df = df1.join(df2, on=['storage'])
    df.write_csv(os.path.join(output_directory, 'storage_built_capacity.csv'))

    # Export summary of storage costs
    costs = pl.DataFrame({'component' : ['cost_per_timepoint_USD', 'cost_per_period_USD', 'total_cost_USD'],
                          'cost' : [  sum( pyo.value(model.eStorCostPerTp[t]) * t.weight for t in system.tp), 
                                            pyo.value(model.eStorCostPerPeriod), 
                                            pyo.value(model.eStorTotalCost)]})
    costs.write_csv(os.path.join(output_directory, 'storage_costs_summary.csv'))

def upload_built_capacities(system, input_directory: str, make_non_expandable: bool = True):
    """Function to upload a previous solution. This can be used to warm start the optimization with a given solution."""
    
    if not os.path.exists(os.path.join(input_directory, "storage_built_capacity.csv")):
        logger.warning(f"No file named 'storage_built_capacity.csv' found in {input_directory}. Skipping upload of built capacities.")
        return
    
    storage_built_capacity = pl.read_csv(os.path.join(input_directory, "storage_built_capacity.csv"), 
                                         schema_overrides={'storage': pl.String, 'built_power_capacity_MW': pl.Float64, 'built_energy_capacity_MWh': pl.Float64})
    storage_built_power_capacity = dict(storage_built_capacity.select("storage", "built_power_capacity_MW").iter_rows())
    storage_built_energy_capacity = dict(storage_built_capacity.select("storage", "built_energy_capacity_MWh").iter_rows())

    storage_to_update = [s for s in system.ess if s.name in storage_built_power_capacity]

    if storage_to_update:
        for stor in storage_to_update:
            stor.cap_existing_energy_MWh += storage_built_energy_capacity[stor.name]
            stor.cap_existing_power_MW += storage_built_power_capacity[stor.name]    
            if make_non_expandable:
                stor.expand_capacity = False
            else:
                stor.expand_capacity = False if stor.cap_existing_power_MW >= stor.cap_max_power_MW else True
    
    logger.info(f"> Updated existing capacities for {len(storage_to_update)} storage units based on input file {os.path.join(input_directory, 'storage_built_capacity.csv')}")