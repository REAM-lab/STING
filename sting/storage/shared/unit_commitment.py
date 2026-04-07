# -----------
# Import python packages
# -----------
import logging
import pyomo.environ as pyo
import os
import polars as pl
from collections import defaultdict

# -----------
# Import sting code
# -----------
from sting.system.core import System
from sting.timescales.core import Timepoint, Scenario
from sting.storage.core import Storage
from sting.utils.runtime_tools import timeit
from sting.modules.capacity_expansion.utils import ModelSettings

# Set up logging
logger = logging.getLogger(__name__)

@timeit
def construct_unit_commitment_model(system: System, model: pyo.ConcreteModel, model_settings: ModelSettings):
    """Construction of storage variables, constraints, and costs."""

    N = system.buses
    T = system.timepoints
    S = system.scenarios
    E = system.storage

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

    logger.info(" - Constraints on storage dispatch")
    if model_settings.single_storage_injection:
        model.cMaxCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                                          (-e.cap_existing_power_MW, m.vDISCHA[e, s, t], +e.cap_existing_power_MW))
    else:
        model.cMaxCharge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vCHARGE[e, s, t] <=  e.cap_existing_power_MW)
    
        model.cMaxDischarge = pyo.Constraint(E, S, T, rule=lambda m, e, s, t: 
                        m.vDISCHA[e, s, t] <=  e.cap_existing_power_MW)
    logger.info(f"   Size: {len(model.cMaxCharge) + len(model.cMaxDischarge)} constraints")

   
    logger.info(" - Constraints on maximum state of charge")
    def max_soc_rule(m: pyo.ConcreteModel, e: Storage, s: Scenario, t: Timepoint):
        if e.cap_existing_energy_MWh > 100:
            return 1e-2 * m.vSOC[e, s, t] <= 1e-2* e.cap_existing_energy_MWh
        elif e.cap_existing_energy_MWh < 10:
            return 10 * m.vSOC[e, s, t] <= 10 * e.cap_existing_energy_MWh
        else:
            return m.vSOC[e, s, t] <= e.cap_existing_energy_MWh

    model.cMaxSOC = pyo.Constraint(E, S, T, rule=max_soc_rule)
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
                                        t.duration_hr * (m.vCHARGE[e, s, t] * e.efficiency_charge 
                                                        - m.vDISCHA[e, s, t] * 1/e.efficiency_discharge) )
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

    # Total storage cost
    model.eStorTotalCost = pyo.Expression(expr = lambda m: sum(m.eStorCostPerTp[t] * t.weight for t in T) )

@timeit
def export_results_unit_commitment(system: System, model: pyo.ConcreteModel, output_directory: str):
    """Storage results to CSV files."""

    # Export discharge and charge results
    df1 = pl.DataFrame([(e.name, s.name, t.name, pyo.value(model.vDISCHA[e, s, t])) for e in system.storage for s in system.scenarios for t in system.timepoints], 
                       columns=['storage', 'scenario', 'timepoint', 'discharge_MW'])
    
    if hasattr(model, 'vCHARGE'):
        df2 = pl.DataFrame([(e.name, s.name, t.name, pyo.value(model.vCHARGE[e, s, t])) for e in system.storage for s in system.scenarios for t in system.timepoints],
                            columns=['storage', 'scenario', 'timepoint', 'charge_MW'])
        df1 = df1.join(df2, on=['storage', 'scenario', 'timepoint'])
    
    # Join with state of charge results
    df3 = pl.DataFrame([(e.name, s.name, t.name, pyo.value(model.vSOC[e, s, t])) for e in system.storage for s in system.scenarios for t in system.timepoints],
                        columns=['storage', 'scenario', 'timepoint', 'state_of_charge_MWh'])
    df1 = df1.join(df3, on=['storage', 'scenario', 'timepoint'])

    df1.write_csv(os.path.join(output_directory, 'storage_dispatch.csv'))

    # Export summary of storage costs
    (pl.DataFrame({'component' : ['cost_per_timepoint_USD', 'total_cost_USD'],
                          'cost' : [  sum( pyo.value(model.eStorCostPerTp[t]) * t.weight for t in system.timepoints), 
                                            pyo.value(model.eStorTotalCost) ]})
        .write_csv(os.path.join(output_directory, 'storage_costs_summary.csv')))

