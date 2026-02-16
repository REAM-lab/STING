# ---------------
# Import python packages
# ---------------
import pyomo.environ as pyo
import numpy as np
import polars as pl
import os
import logging

# ---------------
# Import sting code
# ---------------
from sting.system.core import System
from sting.utils.data_tools import timeit
from sting.modules.power_flow.utils import ModelSettings
from sting.utils.graph_matrices import build_admittance_matrix_from_lines
from sting.bus.utils import load_as_dict

# Set up logging
logger = logging.getLogger(__name__)

@timeit
def construct_ac_power_flow_model(system: System, model: pyo.ConcreteModel, model_settings: ModelSettings):
    """Construction of ac power flow variables, constraints, and costs."""

    N = system.buses
    T = system.timepoints
    L = system.lines
    load = system.loads

    logger.info(" - Processing load data for power flow model")
    model.active_load, model.reactive_load = load_as_dict(load)
    
    logger.info(" - Decision variables of bus voltage magnitudes and angles")
    model.vVMAG = pyo.Var(N, T, 
                             within=pyo.NonNegativeReals,
                             bounds=lambda m, n, t: (n.minimum_voltage_pu, n.maximum_voltage_pu))
    
    model.vANGLE = pyo.Var(N, T, 
                              within=pyo.Reals,
                              bounds= (- np.pi / 2, + np.pi / 2))
    
    logger.info(f"   Size: {len(model.vVMAG) + len(model.vANGLE)} variables")

    if model_settings.load_shedding:
        logger.info(" - Decision variables of active load shedding")
        model.vPSHED = pyo.Var(N, T, within=pyo.NonNegativeReals)
        logger.info(f"   Size: {len(model.vPSHED)} variables")

        logger.info(" - Decision variables of reactive load shedding")
        model.vQSHED = pyo.Var(N, T, within=pyo.NonNegativeReals)
        logger.info(f"   Size: {len(model.vQSHED)} variables")

    slack_bus = next((n for n in N if n.bus_type == 'slack'), None)
    if slack_bus is None:
        slack_bus = N[0]
    model.vANGLE[slack_bus, :].fix(0.0)

    logger.info(" - Power flow per bus expressions")
    Y = build_admittance_matrix_from_lines(len(N), L)
    Ymag = np.abs(Y)
    Yangle = np.angle(Y)
    G = Y.real
    B = Y.imag

    bus_neighbors = {n.id: [N[k] for k in np.nonzero(B[n.id, :])[0]] for n in N}

    match model_settings.power_flow_formulation:
        case 'polar':
            model.ePflowAtBus = pyo.Expression(N, T, 
                        expr=lambda m, n, t: n.base_power_MVA * m.vVMAG[n, t] * sum(m.vVMAG[k, t] * Ymag[n.id, k.id]
                        * pyo.cos(Yangle[n.id, k.id] - m.vANGLE[n, t] + m.vANGLE[k, t]) for k in N
                        ))
            model.eQflowAtBus = pyo.Expression(N, T,
                        expr=lambda m, n, t: -1 * n.base_power_MVA * m.vVMAG[n, t] * sum( m.vVMAG[k, t] * Ymag[n.id, k.id]
                        * pyo.sin(Yangle[n.id, k.id] - m.vANGLE[n, t] + m.vANGLE[k, t]) for k in N
                        ))
        case 'rectangular':
            model.ePflowAtBus = pyo.Expression(N, T,
                        expr=lambda m, n, t: n.base_power_MVA * m.vVMAG[n, t] * sum( m.vVMAG[k, t] 
                                            * ( G[n.id, k.id] * pyo.cos(m.vANGLE[n, t] - m.vANGLE[k, t]) 
                                               + B[n.id, k.id] * pyo.sin(m.vANGLE[n, t] - m.vANGLE[k, t]) ) 
                                               for k in bus_neighbors[n.id]))
            
            model.eQflowAtBus = pyo.Expression(N, T,
                        expr = lambda m, n, t: n.base_power_MVA * m.vVMAG[n, t] * sum( m.vVMAG[k, t]
                                            * (G[n.id, k.id] * pyo.sin(m.vANGLE[n, t] - m.vANGLE[k, t])
                                               - B[n.id, k.id] * pyo.cos(m.vANGLE[n, t] - m.vANGLE[k, t]) )
                                               for k in bus_neighbors[n.id]))

    logger.info(" - Active power balance at each bus")
    model.rescaling_factor_cEnergyBalance = pyo.Param(initialize=1) 
    model.cActivePowerBalance = pyo.Constraint(N, T,
                                         rule=lambda m, n, t: 
                                    (m.eActiveDispatchAtBus[n, t]
                                    + (m.vPSHED[n, t] if ( (model_settings.load_shedding) ) else 0) 
                                    - m.ePflowAtBus[n, t]) * m.rescaling_factor_cEnergyBalance == 
                                    (m.active_load.get((n.name, t.name), 0.0) ) * m.rescaling_factor_cEnergyBalance)
    logger.info(f"   Size: {len(model.cActivePowerBalance)} constraints")

    logger.info(" - Reactive power balance at each bus")
    model.cReactivePowerBalance = pyo.Constraint(N, T,
                                         rule=lambda m, n, t: 
                                    (m.eReactiveDispatchAtBus[n, t]
                                    + (m.vQSHED[n, t] if ( (model_settings.load_shedding) ) else 0) 
                                    - m.eQflowAtBus[n, t]) * m.rescaling_factor_cEnergyBalance == 
                                    (m.reactive_load.get((n.name, t.name), 0.0) ) * m.rescaling_factor_cEnergyBalance)
    logger.info(f"   Size: {len(model.cReactivePowerBalance)} constraints")

    if model_settings.load_shedding:
        logger.info(" - Load shedding cost expressions")
        model.eActiveShedCostPerTp = pyo.Expression(T, rule= lambda m, t: sum(1000 * m.vPSHED[n, t] for n in N)) 
        model.cost_components_per_tp.append(model.eActiveShedCostPerTp)
        logger.info(f"   Size: {len(model.eActiveShedCostPerTp)} expressions")

        logger.info(" - Load shedding cost expressions")
        model.eReactiveShedCostPerTp = pyo.Expression(T, rule= lambda m, t: sum(1000 * m.vQSHED[n, t] for n in N)) 
        model.cost_components_per_tp.append(model.eReactiveShedCostPerTp)
        logger.info(f"   Size: {len(model.eReactiveShedCostPerTp)} expressions")

@timeit
def export_results_ac_power_flow(system: System, model: pyo.ConcreteModel, output_directory: str):
    """Export power flow results to CSV files."""

    N = system.buses
    T = system.timepoints
    L = system.lines

    # Export bus voltage magnitudes and angles
    df = pl.DataFrame( data = [ (n.id, n.name, t.name, pyo.value(model.vVMAG[n, t]), pyo.value(model.vANGLE[n, t] * 180/np.pi)) for n in N for t in T],
                        schema= ['id', 'bus', 'timepoint', 'voltage_magnitude_pu', 'voltage_angle_deg'],
                        orient= 'row')
    df.write_csv(os.path.join(output_directory, 'bus_voltage.csv'))

    # Export active power balance by bus and timepoint
    df = pl.DataFrame(  data = [ (n.name, t.name, 
                                  pyo.value(model.eActiveDispatchAtBus[n, t]),
                                  pyo.value(model.vPSHED[n, t]) if hasattr(model, 'vPSHED') else 0,
                                  model.active_load.get((n.name, t.name), 0.0),
                                  pyo.value(model.ePflowAtBus[n, t]) ) for n in N for t in T],
                        schema= ['bus', 'timepoint', 'generator_dispatch_MW', 'load_shedding_MW', 'load_MW',
                                 'net_line_leaving_flow_MW'],
                        orient= 'row')
    df.write_csv(os.path.join(output_directory, 'active_power_balance_by_bus.csv'))

    # Export reactive power balance by bus and timepoint
    df = pl.DataFrame(  data = [ (n.name, t.name,
                                  pyo.value(model.eReactiveDispatchAtBus[n, t]),
                                  pyo.value(model.vQSHED[n, t]) if hasattr(model, 'vQSHED') else 0,
                                  model.reactive_load.get((n.name, t.name), 0.0),
                                  pyo.value(model.eQflowAtBus[n, t]) ) for n in N for t in T],
                        schema= ['bus', 'timepoint', 'generator_dispatch_MVAR', 'load_shedding_MVAR', 'load_MVAR',
                                 'net_line_leaving_flow_MVAR'],
                        orient= 'row')
    df.write_csv(os.path.join(output_directory, 'reactive_power_balance_by_bus.csv'))

    # Export load shedding results if it is existing
    if hasattr(model, 'vPSHED') and hasattr(model, 'vQSHED'):
        df = pl.DataFrame( data = [ (n.name, t.name, pyo.value(model.vPSHED[n, t]), pyo.value(model.vQSHED[n, t])) for n in N for t in T],
                            schema= ['bus', 'timepoint', 'active_load_shedding_MW', 'reactive_load_shedding_MVAR'],
                            orient= 'row')
        df.write_csv(os.path.join(output_directory, 'load_shedding.csv'))
    
    # Export line flows and losses
    df = pl.DataFrame(  data = [ (l.name, l.base_power_MVA, l.from_bus, l.to_bus, l.r_pu, l.x_pu, l.g_pu, l.b_pu, l.cap_existing_power_MW if l.cap_existing_power_MW is not None else float('inf'),
                                    pyo.value(model.vVMAG[N[l.from_bus_id], t]),
                                    pyo.value(model.vANGLE[N[l.from_bus_id], t]),
                                    pyo.value(model.vVMAG[N[l.to_bus_id], t]),
                                    pyo.value(model.vANGLE[N[l.to_bus_id], t])) for l in L for t in T],
                            schema= ['line', 'base_power_MVA', 'from_bus', 'to_bus', 'r_pu', 'x_pu', 'g_pu', 'b_pu', 'existing_capacity_MW', 
                                     'v_mag_n', 'v_angle_n', 'v_mag_k', 'v_angle_k'],
                            orient= 'row')
    
    df = df.with_columns(
        (pl.col('r_pu') / (pl.col('r_pu')**2 + pl.col('x_pu')**2) + pl.col('g_pu')).alias('G_nn'),
        (- pl.col('x_pu') / (pl.col('r_pu')**2 + pl.col('x_pu')**2) + pl.col('b_pu')).alias('B_nn'),
        (- pl.col('r_pu') / (pl.col('r_pu')**2 + pl.col('x_pu')**2)).alias('G_nk'),
        (+ pl.col('x_pu') / (pl.col('r_pu')**2 + pl.col('x_pu')**2)).alias('B_nk'))
    
    def compute_line_ac_power_flow(base_power_MVA, v_mag_n, v_angle_n, v_mag_k, v_angle_k, G_nn, B_nn, G_nk, B_nk):
        delta = v_angle_n - v_angle_k
        P = base_power_MVA * v_mag_n * ( v_mag_n * G_nn + v_mag_k * (G_nk * np.cos(delta) + B_nk * np.sin(delta)) )
        Q = base_power_MVA * v_mag_n * ( -1 * v_mag_n * B_nn + v_mag_k * (G_nk * np.sin(delta) - B_nk * np.cos(delta)) )
        return P, Q
    
    P, Q = compute_line_ac_power_flow(df['base_power_MVA'],  df['v_mag_n'], df['v_angle_n'], df['v_mag_k'], df['v_angle_k'],
                                     df['G_nn'], df['B_nn'], df['G_nk'], df['B_nk'])
    df = df.with_columns(
        pl.Series(name='active_power_from_bus_MW', values=P),
        pl.Series(name='reactive_power_from_bus_MVAR', values=Q)
    )

    P, Q = compute_line_ac_power_flow(df['base_power_MVA'],  df['v_mag_k'], df['v_angle_k'], df['v_mag_n'], df['v_angle_n'],
                                     df['G_nn'], df['B_nn'], df['G_nk'], df['B_nk'])
    
    df = df.with_columns(
        pl.Series(name='active_power_to_bus_MW', values=P),
        pl.Series(name='reactive_power_to_bus_MVAR', values=Q)
    )

    df = df.with_columns(
        (pl.col('active_power_from_bus_MW') + pl.col('active_power_to_bus_MW')).alias('active_power_loss_MW'),
        (pl.col('reactive_power_from_bus_MVAR') + pl.col('reactive_power_to_bus_MVAR')).alias('reactive_power_loss_MVAR')
    )

    df = df.rename({'v_mag_n': 'from_bus_mag_pu', 'v_angle_n': 'from_bus_angle_rad', 'v_mag_k': 'to_bus_mag_pu', 'v_angle_k': 'to_bus_angle_rad'})
    df = df.select(['line', 'from_bus', 'to_bus', 'existing_capacity_MW', 'active_power_from_bus_MW', 'reactive_power_from_bus_MVAR', 'active_power_to_bus_MW', 'reactive_power_to_bus_MVAR',
                     'active_power_loss_MW', 'reactive_power_loss_MVAR'])
    
    df.write_csv(os.path.join(output_directory, 'line_flows.csv'))
