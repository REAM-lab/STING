# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import pyomo.environ as pyo
import numpy as np
import polars as pl
import os

# -------------
# Import sting code
# --------------
from sting.timescales.core import Timepoint, Scenario
from sting.utils.graph_matrices import build_admittance_matrix_from_lines
from sting.utils.dynamical_systems import DynamicalVariables


# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Bus:
    id: int = field(default=-1, init=False)
    name: str
    bus_type: str = None
    sbase: float = None
    vbase: float = None
    fbase: float = None
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

    model.vTHETA[slack_bus.id, :, :].fix(0.0)

    model.eFlowAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 100 * sum(B[n.id, k.id] * (m.vTHETA[n, s, t] - m.vTHETA[k, s, t]) for k in N))
    
    def cMaxFlowPerLine_rule(m, l, s, t):
        if l.rating_MVA > 0:
            return (-l.rating_MVA,
                    100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * 
                    (m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t]),
                    l.rating_MVA)
        else:
            return pyo.Constraint.Skip
    
    model.cMaxFlowPerLine = pyo.Constraint(L, S, T, rule=cMaxFlowPerLine_rule)
    
    def cDiffAngle_rule(m, l, s, t):
        if (l.angle_min_deg > -360) and (l.angle_max_deg < 360):
            return (l.angle_min_deg * np.pi / 180, 
                    m.vTHETA[N[l.from_bus_id], s, t] - m.vTHETA[N[l.to_bus_id], s, t],
                    l.angle_max_deg * np.pi / 180)
        else:
            return pyo.Constraint.Skip
        
    model.cDiffAngle = pyo.Constraint(L, S, T, rule=cDiffAngle_rule)

    # Power balance at each bus
    model.cPowerBalance = pyo.Constraint(N, S, T,
                                         rule=lambda m, n, s, t: 
                            m.eGenAtBus[n, s, t] + m.eNetDischargeAtBus[n, s, t] >= 
                            next((ld.load_MW for ld in load 
                                    if  ((ld.bus == n.name) and 
                                        (ld.scenario == s.name) and 
                                        (ld.timepoint == t.name))), 0.0) + m.eFlowAtBus[n, s, t]
                            
                            )

def export_results_capacity_expansion(system, model: pyo.ConcreteModel, output_directory: str):
    
    dct = model.vTHETA.extract_values()

    def dct_to_tuple(dct_item):
        k, v = dct_item
        bus, sc, t = k
        return (bus.name, sc.name, t.name, v * 180 / np.pi) 
    
    df = pl.DataFrame(  
                        schema =['bus', 'scenario', 'timepoint', 'voltage_angle_rad'],
                        data= map(dct_to_tuple, dct.items()) )

    df.write_csv(os.path.join(output_directory, 'bus_voltage_angles.csv'))
                        