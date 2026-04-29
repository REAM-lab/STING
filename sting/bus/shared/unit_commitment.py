# ---------------
# Import python packages
# ---------------
import pyomo.environ as pyo
import numpy as np
import polars as pl
import logging
import os

# ---------------
# Import sting code
# ---------------
from sting.system.core import System
from sting.modules.capacity_expansion.utils import ModelSettings
from sting.utils.runtime_tools import timeit
from sting.utils.graph_matrices import build_admittance_matrix_from_lines
from sting.bus.utils import load_as_dict

# Set up logging
logger = logging.getLogger(__name__)

@timeit    
def construct_unit_commitment_model(system: System, model: pyo.ConcreteModel, model_settings: ModelSettings):
    """Construction of transmission variables, constraints, and costs for unit commitment model."""

    N = system.buses
    T = system.timepoints
    S = system.scenarios
    L = system.lines
    load = system.loads

    logger.info(" - Load data processing")
    model.load_lookup = load_as_dict(load)[0]
    load_buses = {ld.bus for ld in load if np.abs(ld.load_MW) != 0}
    N_load = [n for n in N if n.name in load_buses]

    if model_settings.load_shedding:
        logger.info(" - Decision variables of load shedding")
        model.vSHED = pyo.Var(N_load, S, T, within=pyo.NonNegativeReals)
        logger.info(f"   Size: {len(model.vSHED)} variables")

    if model_settings.power_flow == 'dc':    
        logger.info(" - Decision variables of bus angles")
        model.vTHETA = pyo.Var(N, S, T, within=pyo.Reals)
        logger.info(f"   Size: {len(model.vTHETA)} variables")

        slack_bus = next((n for n in N if n.bus_type == 'slack'), None)
        if slack_bus is None:
            slack_bus = N[0]
        model.vTHETA[slack_bus, :, :].fix(0.0)

        logger.info(" - Power flow per bus expressions")
        Y = build_admittance_matrix_from_lines(len(N), L)
        B = Y.imag

        N_at_bus = {n.id: [N[k] for k in np.nonzero(B[n.id, :])[0] if k != n.id] for n in N}
        model.eFlowAtBus = pyo.Expression(N, S, T, expr=lambda m, n, s, t: 100 * pyo.quicksum(B[n.id, k.id] * (m.vTHETA[n, s, t] - m.vTHETA[k, s, t]) for k in N_at_bus[n.id]) )

    elif model_settings.power_flow == 'transport':
        logger.info(" - Decision variables of line flows")
        # vFLOW_SENT_AT_FROM_BUS is the flow leaving from_bus going to to_bus
        model.vFLOW_SENT_AT_FROM_BUS = pyo.Var(L, S, T, within=pyo.NonNegativeReals)
        # vFLOW_SENT_AT_TO_BUS is the flow leaving at to_bus going to from_bus
        model.vFLOW_SENT_AT_TO_BUS = pyo.Var(L, S, T, within=pyo.NonNegativeReals)
        logger.info(f"   Size: {len(model.vFLOW_SENT_AT_FROM_BUS) + len(model.vFLOW_SENT_AT_TO_BUS)} variables")
        
        # L_from_bus is the set of lines for which bus n is a from_bus
        L_from_bus = {n.id: [l for l in L if l.from_bus == n.name] for n in N}
        # L_to_bus is the set of lines for which bus n is a to_bus
        L_to_bus = {n.id: [l for l in L if l.to_bus == n.name] for n in N}

        model.eFlowAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 
                                          pyo.quicksum(m.vFLOW_SENT_AT_FROM_BUS[l, s, t] - l.efficiency * m.vFLOW_SENT_AT_TO_BUS[l, s, t] for l in L_from_bus[n.id]) 
                                          + pyo.quicksum(m.vFLOW_SENT_AT_TO_BUS[l, s, t] - l.efficiency * m.vFLOW_SENT_AT_FROM_BUS[l, s, t] for l in L_to_bus[n.id]) )
        
        
    L_cap_constrained = {l for l in L if l.cap_existing_power_MW is not None}
    logger.info(" - Maximum flow constraints per line")
    if model_settings.power_flow == 'dc':
        def cFlowPerNonExpLine_rule(m, l, s, t):
                b = l.x_pu / (l.x_pu**2 + l.r_pu**2)
                return  (-1/100 * l.cap_existing_power_MW,  b * (m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t]), 1/100 * l.cap_existing_power_MW)
            
        model.cFlowPerNonExpLine = pyo.Constraint(L_cap_constrained, S, T, rule=cFlowPerNonExpLine_rule)
        logger.info(f"   Size: {len(model.cFlowPerNonExpLine)} constraints")

    if model_settings.power_flow == 'transport':
        #model.cFlowPerNonExpLine1 = pyo.Constraint(L_cap_constrained, S, T, rule=lambda m, l, s, t: m.vFLOW_SENT_AT_FROM_BUS[l, s, t] <= l.cap_existing_power_MW)
        #model.cFlowPerNonExpLine2 = pyo.Constraint(L_cap_constrained, S, T, rule=lambda m, l, s, t: m.vFLOW_SENT_AT_TO_BUS[l, s, t] <= l.cap_existing_power_MW)
        model.cFlowPerNonExpLine3 = pyo.Constraint(L_cap_constrained, S, T, rule=lambda m, l, s, t: 1e-2*(m.vFLOW_SENT_AT_TO_BUS[l, s, t] + m.vFLOW_SENT_AT_FROM_BUS[l, s, t]) <= 1e-2*l.cap_existing_power_MW)
            
        logger.info(f"   Size: {len(model.cFlowPerNonExpLine3)} constraints")

    if model_settings.angle_difference_limits and model_settings.power_flow == 'dc':

        logger.info(" - Angle difference limit constraints per line")
        lines_with_angle_limits = {l for l in L if (l.angle_min_deg > -360) and (l.angle_max_deg < 360)}
        model.cDiffAngle = pyo.Constraint(lines_with_angle_limits, S, T, rule= lambda m, l, s, t: 
                                                                                (l.angle_min_deg * np.pi / 180, 
                                                                                m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t],
                                                                                l.angle_max_deg * np.pi / 180))
        logger.info(f"   Size: {len(model.cDiffAngle)} constraints")

    logger.info(" - Energy balance at each bus")
    if model_settings.power_flow == 'dc':
        model.rescaling_factor_cEnergyBalance = pyo.Param(initialize=1e-2)
    else:
        model.rescaling_factor_cEnergyBalance = pyo.Param(initialize=1.0) 
    model.cEnergyBalance = pyo.Constraint(N, S, T,
                                         rule=lambda m, n, s, t: 
                            (m.eGenAtBus[n, s, t] 
                             + m.eNetDischargeAtBus[n, s, t] 
                             + (m.vSHED[n, s, t] if ((model_settings.load_shedding) and (n.name in load_buses)) else 0) 
                             - m.eFlowAtBus[n, s, t]) * model.rescaling_factor_cEnergyBalance == 
                            (model.load_lookup.get((n.name, s.name, t.name), 0.0) ) * model.rescaling_factor_cEnergyBalance)
                            
    logger.info(f"   Size: {len(model.cEnergyBalance)} constraints")

    if model_settings.load_shedding:
        logger.info(" - Load shedding cost expressions")
        model.eShedCostPerTp = pyo.Expression(T, rule= lambda m, t: 1/len(S) * sum(s.probability * 5000 * m.vSHED[n, s, t] for n in N_load for s in S)) 
        model.cost_components_per_tp.append(model.eShedCostPerTp)
        logger.info(f"   Size: {len(model.eShedCostPerTp)} expressions")

        model.eShedTotalCost = pyo.Expression(
                                expr =  lambda m: sum(m.eShedCostPerTp[t] * t.weight for t in T)
                                )