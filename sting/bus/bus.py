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

# -------------
# Import sting code
# --------------
from sting.timescales.core import Timepoint, Scenario
from sting.utils.graph_matrices import build_admittance_matrix_from_lines
from sting.utils.dynamical_systems import DynamicalVariables
from sting.utils.data_tools import pyovariable_to_df, pyodual_to_df


# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Bus:
    id: int = field(default=-1, init=False)
    name: str
    bus_type: str = None
    base_power_MVA: float = None
    base_voltage_kV: float = None
    base_frequency_Hz: float = None
    max_flow_MW: float = None
    v_min: float = None
    v_max: float = None
    p_load: float = None
    q_load: float = None
    tags: ClassVar[list[str]] = []

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return hash(self.id)

    def __repr__(self):
        return f"Bus(id={self.id}, bus='{self.name}')"
    
    def post_system_init(self, system):
        # Exit if the bus has not constraint on max power flow
        if self.max_flow_MW is not None:
            return
        
        # Otherwise, we deduce max power flow based on the lines
        # the current bus is connected to.
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

    
def construct_capacity_expansion_model(system, model: pyo.ConcreteModel, model_settings: dict):

    N = system.bus
    T = system.tp
    S = system.sc
    L = system.line_pi
    load = system.load

    model.vTHETA = pyo.Var(N, S, T, within=pyo.Reals)
    
    slack_bus = next(n for n in N if n.bus_type == 'slack')

    Y = build_admittance_matrix_from_lines(len(N), L)
    B = Y.imag

    model.vTHETA[slack_bus, :, :].fix(0.0)

    N_at_bus = {n.id: [N[k] for k in np.nonzero(B[n.id, :])[0] if k != n.id] for n in N}

    model.eFlowAtBus = pyo.Expression(N, S, T, expr=lambda m, n, s, t: 100 * quicksum(B[n.id, k.id] * (m.vTHETA[n, s, t] - m.vTHETA[k, s, t]) for k in N_at_bus[n.id]) )

    
    if model_settings["consider_line_capacity"] == True:

        L_expandable = set([l for l in L if (l.expand_capacity == True and l.cap_existing_power_MW is not None)])
        L_nonexpandable = set([l for l in L if (l.expand_capacity == False and l.cap_existing_power_MW is not None)])
        
        model.vCAPL = pyo.Var(L_expandable, within=pyo.NonNegativeReals)

        def cMaxFlowPerExpLine_rule(m, l, s, t):
                return  100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * (m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t]) <= m.vCAPL[l] + l.cap_existing_power_MW
        
        def cMinFlowPerExpLine_rule(m, l, s, t):
                return  100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * (m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t]) >= -(m.vCAPL[l] + l.cap_existing_power_MW)
   
        model.cMaxFlowPerExpLine = pyo.Constraint(L_expandable, S, T, rule=cMaxFlowPerExpLine_rule)
        model.cMinFlowPerExpLine = pyo.Constraint(L_expandable, S, T, rule=cMinFlowPerExpLine_rule)

        def cFlowPerNonExpLine_rule(m, l, s, t):
                return  (-l.cap_existing_power_MW,  100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * (m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t]), l.cap_existing_power_MW)
        
        model.cFlowPerNonExpLine = pyo.Constraint(L_nonexpandable, S, T, rule=cFlowPerNonExpLine_rule)
        
        model.eLineCostPerPeriod = pyo.Expression(expr = lambda m: sum(l.cost_fixed_power_USDperkW * m.vCAPL[l] * 1000 for l in L_expandable))

    if model_settings["consider_bus_max_flow"] == True:

        buses_with_max_flow = set([n for n in N if n.max_flow_MW is not None])
        model.cFlowPerBus = pyo.Constraint(buses_with_max_flow, S, T, rule=lambda m, n, s, t: (-n.max_flow_MW, m.eFlowAtBus[n, s, t], n.max_flow_MW))

    if model_settings["consider_angle_limits"] == True:

        lines_with_angle_limits = set([l for l in L if (l.angle_min_deg > -360) and (l.angle_max_deg < 360)])
        model.cDiffAngle = pyo.Constraint(lines_with_angle_limits, S, T, rule= lambda m, l, s, t: 
                                                                                (l.angle_min_deg * np.pi / 180, 
                                                                                m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t],
                                                                                l.angle_max_deg * np.pi / 180))

    # Power balance at each bus
    load_lookup = {(ld.bus, ld.scenario, ld.timepoint): ld.load_MW for ld in load}
    model.cEnergyBalance = pyo.Constraint(N, S, T,
                                         rule=lambda m, n, s, t: 
                            (m.eGenAtBus[n, s, t] + m.eNetDischargeAtBus[n, s, t]) * t.weight == 
                            (load_lookup.get((n.name, s.name, t.name), 0.0) + m.eFlowAtBus[n, s, t]) * t.weight
                            )
    

def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):

    # Export line capacities 
    if hasattr(model, 'vCAPL'):
        pyovariable_to_df(model.vCAPL, 
                            dfcol_to_field={'line': 'name'}, 
                            value_name='capacity_MW', 
                            csv_filepath=os.path.join(output_directory, 'line_built_capacity.csv'))
        costs = pl.DataFrame({ 'component' : ['CostPerPeriod_USD'],
                                'cost' : [  pyo.value(model.eLineCostPerPeriod)]})
        costs.write_csv(os.path.join(output_directory, 'line_costs_summary.csv'))
    
    # Export LMPs
    df = pyodual_to_df(model.dual, model.cEnergyBalance, 
                            dfcol_to_field={'bus': 'name', 'scenario': 'name', 'timepoint': 'name'}, 
                            value_name='local_marginal_price_USDperMWh')
    
    df = df.with_columns(
        (pl.col('local_marginal_price_USDperMWh') / model.rescaling_factor_obj).alias('local_marginal_price_USDperMWh'))
    
    df.write_csv(os.path.join(output_directory, 'local_marginal_prices.csv'))

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

    
                        