# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import pyomo.environ as pyo
import numpy as np
import polars as pl
import os
from pyomo.environ import quicksum
import logging

# -------------
# Import sting code
# --------------
from sting.timescales.core import Timepoint, Scenario
from sting.utils.graph_matrices import build_admittance_matrix_from_lines
from sting.utils.dynamical_systems import DynamicalVariables
from sting.utils.data_tools import pyovariable_to_df, pyodual_to_df, timeit

logger = logging.getLogger(__name__)

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Bus:
    id: int = field(default=-1, init=False)
    name: str
    bus_type: str = None
    zone: str = None
    kron_removable_bus: bool = None
    base_power_MVA: float = None
    base_voltage_kV: float = None
    base_frequency_Hz: float = None
    max_flow_MW: float = None
    v_min: float = None
    v_max: float = None
    p_load: float = 0.0
    q_load: float = 0.0
    tags: ClassVar[list[str]] = []

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id

    def __repr__(self):
        return f"Bus(id={self.id}, bus='{self.name}')"
    
    def post_system_init(self, system):
        # We deduce max power flow based on the lines
        # the current bus is connected to.
        if self.max_flow_MW is not None:
            return  # Already defined
        
        self.max_flow_MW = 0.0
        connected_lines = {line for line in system.line_pi if (self.name in [line.from_bus, line.to_bus])}

        for line in connected_lines:
            # There is no constraint on max power flow on the line,
            # thus the bus should also inherit no constraint (and we can exit).
            if (line.cap_existing_power_MW is None):
                self.max_flow_MW = None
                return
            # Otherwise
            else:
                self.max_flow_MW += line.cap_existing_power_MW

@dataclass(slots=True)
class Load:
    id: int = field(default=-1, init=False)
    bus: str
    scenario: str
    timepoint: str
    load_MW: float

@timeit    
def construct_capacity_expansion_model(system, model: pyo.ConcreteModel, model_settings: dict, kron_variables):
    """Construction of transmission variables, constraints, and costs for capacity expansion model."""

    N = system.bus
    T = system.tp
    S = system.sc
    L = system.line_pi
    load = system.load


    logger.info(" - Load data processing")
    load_df = pl.DataFrame(
                        schema = ['id', 'bus', 'scenario', 'timepoint', 'load_MW'],
                        data= map(lambda ld: (ld.id, ld.bus, ld.scenario, ld.timepoint, ld.load_MW), load)
                        )
    if len(load_df.select(['bus', 'scenario', 'timepoint']).unique()) != (load_df.height):
        logger.info("There are multiple load entries for the same bus, scenario, and timepoint. They will be summed.")
        load_df = load_df.group_by(['bus', 'scenario', 'timepoint']).agg(pl.col('load_MW').sum().alias('load_MW'))
    
    load_lookup = {(ld['bus'], ld['scenario'], ld['timepoint']): ld['load_MW'] for ld in load_df.iter_rows(named=True)}
    load_buses = {ld.bus for ld in load if np.abs(ld.load_MW) != 0}
    N_load = [n for n in N if n.name in load_buses]

    logger.info(" - Decision variables of bus angles")
    model.vTHETA = pyo.Var(N, S, T, within=pyo.Reals)
    logger.info(f"   Size: {len(model.vTHETA)} variables")

    if model_settings.load_shedding:
        logger.info(" - Decision variables of load shedding")
        model.vSHED = pyo.Var(N_load, S, T, within=pyo.NonNegativeReals)
        logger.info(f"   Size: {len(model.vSHED)} variables")
    
    slack_bus = next((n for n in N if n.bus_type == 'slack'), None)
    if slack_bus is None:
            slack_bus = N[0]
    model.vTHETA[slack_bus, :, :].fix(0.0)

    logger.info(" - Power flow per bus expressions")
    Y = build_admittance_matrix_from_lines(len(N), L)
    B = Y.imag

    N_at_bus = {n.id: [N[k] for k in np.nonzero(B[n.id, :])[0] if k != n.id] for n in N}

    model.eFlowAtBus = pyo.Expression(N, S, T, expr=lambda m, n, s, t: 100 * quicksum(B[n.id, k.id] * (m.vTHETA[n, s, t] - m.vTHETA[k, s, t]) for k in N_at_bus[n.id]) )

    if model_settings.line_capacity_expansion:

        logger.info(" - Set of expandable and non-expandable lines")
        L_expandable = {l for l in L if (l.expand_capacity == True and l.cap_existing_power_MW is not None)}
        L_nonexpandable = {l for l in L if (l.expand_capacity == False and l.cap_existing_power_MW is not None)}
        
        logger.info(" - Decision variables of line capacity expansion")
        model.vCAPL = pyo.Var(L_expandable, within=pyo.NonNegativeReals)
        logger.info(f"   Size: {len(model.vCAPL)} variables")

        logger.info(" - Maximum and minimum flow constraints per expandable line")
        def cMaxFlowPerExpLine_rule(m, l, s, t):
                return  100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * (m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t]) <= m.vCAPL[l] + l.cap_existing_power_MW
        
        def cMinFlowPerExpLine_rule(m, l, s, t):
                return  100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * (m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t]) >= -(m.vCAPL[l] + l.cap_existing_power_MW)
   
        model.cMaxFlowPerExpLine = pyo.Constraint(L_expandable, S, T, rule=cMaxFlowPerExpLine_rule)
        model.cMinFlowPerExpLine = pyo.Constraint(L_expandable, S, T, rule=cMinFlowPerExpLine_rule)

        logger.info(" - Maximum and minimum flow constraints per non-expandable line")
        def cFlowPerNonExpLine_rule(m, l, s, t):
                return  (-l.cap_existing_power_MW,  100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * (m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t]), l.cap_existing_power_MW)
        
        model.cFlowPerNonExpLine = pyo.Constraint(L_nonexpandable, S, T, rule=cFlowPerNonExpLine_rule)
        logger.info(f"   Size: {len(model.cMaxFlowPerExpLine) + len(model.cMinFlowPerExpLine) + len(model.cFlowPerNonExpLine)} constraints")
        
        logger.info(" - Line cost per period expression")
        model.eLineCostPerPeriod = pyo.Expression(expr = lambda m: sum(l.cost_fixed_power_USDperkW * m.vCAPL[l] * 1000 for l in L_expandable))

        model.cost_components_per_period.append(model.eLineCostPerPeriod)

    elif (model_settings.line_capacity):
        
        logger.info(" - Maximum and minimum flow constraints per line")
        L_cap_constrained = {l for l in L if l.cap_existing_power_MW is not None}
        def cFlowPerNonExpLine_rule(m, l, s, t):
                return  (-l.cap_existing_power_MW,  100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * (m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t]), l.cap_existing_power_MW)
        
        model.cFlowPerNonExpLine = pyo.Constraint(L_cap_constrained, S, T, rule=cFlowPerNonExpLine_rule)
        logger.info(f"   Size: {len(model.cFlowPerNonExpLine)} constraints")

    if model_settings.kron_equivalent_flow_constraints and (model_settings.line_capacity == False):

        logger.info(" - Constraints for Kron system flow calculations")
        X_2 = -1 * kron_variables.invB_qq @ kron_variables.B_qp # (q, p)
        X_1 = np.eye(X_2.shape[1])
        X = np.vstack((X_1, X_2))

        N_original = kron_variables.original_system.bus # p + q
        L_original = kron_variables.original_system.line_pi
        N_at_bus_original = {n.id: [N_original[k] for k in np.nonzero(X[n.id, :])[0]] for n in N_original}

        model.eTheta = pyo.Expression(
            N_original, S, T, 
            expr=lambda m, n, s, t: quicksum(X[n.id, k.id] * m.vTHETA[k, s, t] for k in N_at_bus_original[n.id]) )
        
        def cFlowPerNonExpLine_rule(m, l, s, t):
                b = 100 * l.x_pu / (l.x_pu**2 + l.r_pu**2)
                max_flow = l.cap_existing_power_MW
                i = N_original[l.from_bus_id]
                j = N_original[l.to_bus_id]

                return  (-max_flow, b * (m.eTheta[i, s, t] - m.eTheta[j, s, t]), max_flow)
        
        model.cFlowPerNonExpLine = pyo.Constraint(L_original, S, T, rule=cFlowPerNonExpLine_rule)
         

    buses_with_max_flow = {n for n in N if n.max_flow_MW is not None}
    if model_settings.bus_max_flow_expansion:
         
        logger.info(" - Maximum flow variables per bus")
        model.vCAPBus = pyo.Var(buses_with_max_flow, within=pyo.NonNegativeReals)
        logger.info(f"   Size: {len(model.vCAPBus)} variables")

        logger.info(" - Maximum flow constraints per bus")
        model.cMaxFlowPerBus = pyo.Constraint(buses_with_max_flow, S, T, rule=lambda m, n, s, t: m.eFlowAtBus[n, s, t] <= n.max_flow_MW + m.vCAPBus[n])
        model.cMinFlowPerBus = pyo.Constraint(buses_with_max_flow, S, T, rule=lambda m, n, s, t: m.eFlowAtBus[n, s, t] >= -(n.max_flow_MW + m.vCAPBus[n]))
        logger.info(f"   Size: {len(model.cMinFlowPerBus) + len(model.cMaxFlowPerBus)} constraints")

        logger.info(" - Bus flow expansion cost expressions")
        model.eBusFlowExpCostPerPeriod = pyo.Expression(expr = lambda m: sum(1000 * 150 * m.vCAPBus[n] for n in buses_with_max_flow))
        model.cost_components_per_period.append(model.eBusFlowExpCostPerPeriod)

    elif model_settings.bus_max_flow:

        logger.info(" - Maximum flow constraints per bus")
        model.cFlowPerBus = pyo.Constraint(buses_with_max_flow, S, T, rule=lambda m, n, s, t: (-n.max_flow_MW, m.eFlowAtBus[n, s, t], n.max_flow_MW))
        logger.info(f"   Size: {len(model.cFlowPerBus)} constraints")

    if model_settings.angle_difference_limits:

        logger.info(" - Angle difference limit constraints per line")
        lines_with_angle_limits = {l for l in L if (l.angle_min_deg > -360) and (l.angle_max_deg < 360)}
        model.cDiffAngle = pyo.Constraint(lines_with_angle_limits, S, T, rule= lambda m, l, s, t: 
                                                                                (l.angle_min_deg * np.pi / 180, 
                                                                                m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t],
                                                                                l.angle_max_deg * np.pi / 180))
        logger.info(f"   Size: {len(model.cDiffAngle)} constraints")

    logger.info(" - Energy balance at each bus")
    model.cEnergyBalance = pyo.Constraint(N, S, T,
                                         rule=lambda m, n, s, t: 
                            (m.eGenAtBus[n, s, t] 
                             + m.eNetDischargeAtBus[n, s, t] 
                             + (m.vSHED[n, s, t] if ((model_settings.load_shedding) and (n.name in load_buses)) else 0) )  == 
                            (load_lookup.get((n.name, s.name, t.name), 0.0) + m.eFlowAtBus[n, s, t]) 
                            )
    logger.info(f"   Size: {len(model.cEnergyBalance)} constraints")

    if model_settings.load_shedding:
        logger.info(" - Load shedding cost expressions")
        model.eShedCostPerTp = pyo.Expression(T, rule= lambda m, t: 1/len(S) * sum(s.probability * 5000 * m.vSHED[n, s, t] for n in N_load for s in S)) 
        model.cost_components_per_tp.append(model.eShedCostPerTp)
        logger.info(f"   Size: {len(model.eShedCostPerTp)} expressions")

        model.eShedTotalCost = pyo.Expression(
                                expr =  lambda m: sum(m.eShedCostPerTp[t] * t.weight for t in T)
                                )
    

@timeit
def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):
    """Transmission results to CSV files."""

    # Export load shedding results if it is existing
    if hasattr(model, 'vSHED'):
        pyovariable_to_df(model.vSHED, 
                          dfcol_to_field={'bus': 'name', 'scenario': 'name', 'timepoint': 'name'}, 
                          value_name='load_shedding_MW', 
                          csv_filepath=os.path.join(output_directory, 'load_shedding.csv'))
        
        # Export summary of load shedding costs
        costs = pl.DataFrame({'component' : ['TotalCost_USD'],
                              'cost' : [  pyo.value(model.eShedTotalCost)]})
        costs.write_csv(os.path.join(output_directory, 'load_shedding_costs_summary.csv'))

    # Export line capacities 
    if hasattr(model, 'vCAPL'):
        pyovariable_to_df(model.vCAPL, 
                            dfcol_to_field={'line': 'name'}, 
                            value_name='built_capacity_MW', 
                            csv_filepath=os.path.join(output_directory, 'line_built_capacity.csv'))
        costs = pl.DataFrame({ 'component' : ['CostPerPeriod_USD'],
                                'cost' : [  pyo.value(model.eLineCostPerPeriod)]})
        costs.write_csv(os.path.join(output_directory, 'line_costs_summary.csv'))
    
    # Export bus max flow expansions
    if hasattr(model, 'vCAPBus'):
        pyovariable_to_df(model.vCAPBus, 
                            dfcol_to_field={'bus': 'name'}, 
                            value_name='built_capacity_MW', 
                            csv_filepath=os.path.join(output_directory, 'bus_built_capacity.csv'))
        costs = pl.DataFrame({ 'component' : ['CostPerPeriod_USD'],
                                'cost' : [  pyo.value(model.eBusFlowExpCostPerPeriod)]})
        costs.write_csv(os.path.join(output_directory, 'bus_max_flow_costs_summary.csv'))

    # Export LMPs
    try:
        df = pyodual_to_df(model.dual, model.cEnergyBalance, 
                            dfcol_to_field={'bus': 'name', 'scenario': 'name', 'timepoint': 'name'}, 
                            value_name='local_marginal_price_USDperMWh')
        
        df_timepoints = pl.DataFrame(
                        schema = ['timepoint', 'weight'],
                        data= map(lambda t: (t.name, t.weight), system.tp)
                        )
        
        df = df.join(df_timepoints, left_on='timepoint', right_on='timepoint')
    
        df = df.with_columns(
            (pl.col('local_marginal_price_USDperMWh') / (model.rescaling_factor_obj * pl.col('weight'))).alias('local_marginal_price_USDperMWh'))
    
        df.write_csv(os.path.join(output_directory, 'local_marginal_prices.csv'))
    except:
        df = pl.DataFrame(data=None, schema=["bus", "scenario", "timepoint", "local_marginal_price_USDperMWh"])
        df.write_csv(os.path.join(output_directory, 'local_marginal_prices.csv'))
        logger.warning("Could not export LMPs to CSV file. Solver is not supported or duals not available.")

    # Export line flows and losses   
    dct = model.vTHETA.extract_values()

    def dct_to_tuple(dct_item):
        k, v = dct_item
        bus, sc, t = k
        return (bus.name, sc.name, t.name, v) 
    
    df_angle = pl.DataFrame(  
                        schema =['bus', 'scenario', 'timepoint', 'angle_rad'],
                        data= map(dct_to_tuple, dct.items()) )

    df_line = pl.DataFrame(
        schema = ['name', 'from_bus', 'to_bus', 'r_pu', 'x_pu', 'g_pu', 'b_pu'],
        data= map(lambda l: (l.name, l.from_bus, l.to_bus, l.r_pu, l.x_pu, l.g_pu, l.b_pu), system.line_pi)
    )

    # Join
    df = df_line.join(df_angle,
                        left_on = ['from_bus'],
                        right_on = ['bus'],
                        how = 'right')
    
    df = df.drop_nulls()
    df = df.rename({'angle_rad': 'from_bus_angle_rad', 'bus': 'from_bus'})

    # Join again
    df = df.join(df_angle, 
                 left_on = ['to_bus', 'scenario', 'timepoint'],
                 right_on = ['bus', 'scenario', 'timepoint'],
                 how = 'right')
    df = df.drop_nulls()
    df = df.rename({'angle_rad': 'to_bus_angle_rad', 'bus': 'to_bus'})
    
    # Compute admittance
    df = df.with_columns(
        (pl.col('x_pu') / (pl.col('r_pu')**2 + pl.col('x_pu')**2)).alias('y_pu'))
    
    # Compute DC flow
    df = df.with_columns(
        (100 * pl.col('y_pu') * (pl.col('from_bus_angle_rad') - pl.col('to_bus_angle_rad'))).alias('DCflow_MW'))
    
    # Compute losses
    df = df.with_columns(
        (pl.col('r_pu') * pl.col('DCflow_MW')**2).alias('losses_MW'))
    
    # Transform radians to degrees
    df = df.with_columns(
        (pl.col('from_bus_angle_rad') * 180 / np.pi).alias('from_bus_angle_deg'))
    df = df.with_columns(
        (pl.col('to_bus_angle_rad') * 180 / np.pi).alias('to_bus_angle_deg'))
    
    # Select columns to export
    df = df.select([
                    'name', 'from_bus', 'to_bus', 'scenario', 'timepoint', 'from_bus_angle_deg', 'to_bus_angle_deg',
                    'DCflow_MW', 'losses_MW'])
    
    # Export to CSV
    df.write_csv(os.path.join(output_directory, 'line_flows.csv'))

    
                        