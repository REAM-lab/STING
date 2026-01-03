# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import pyomo.environ as pyo
import numpy as np


# -------------
# Import sting code
# --------------
from sting.timescales.core import Timepoint, Scenario
from sting.utils.graph_matrices import build_admittance_matrix2

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Bus:
    idx: int = field(default=-1, init=False)
    bus: str
    bus_type: str = None
    sbase: float = None
    vbase: float = None
    fbase: float = None
    v_min: float = None
    v_max: float = None
    p_load: float = None
    q_load: float = None
    tags: ClassVar[list[str]] = []

    def __repr__(self):
        return f"Bus(idx={self.idx}, bus='{self.bus}')"

    def __hash__(self):
        return hash(self.idx)

  
@dataclass(slots=True)
class Load:
    idx: int = field(default=-1, init=False)
    bus: str
    scenario: str
    timepoint: str
    load_MW: float
    bus_idx: int = field(default=None, init=False)
    scenario_idx: int = field(default=None, init=False)
    timepoint_idx: int = field(default=None, init=False)

    def assign_indices(self, buses: list[Bus], scenarios: list[Scenario], timepoints: list[Timepoint]):
        self.bus_idx = next(filter(lambda b: b.bus == self.bus, buses)).idx
        self.scenario_idx = next(filter(lambda s: s.scenario == self.scenario, scenarios)).idx
        self.timepoint_idx = next(filter(lambda t: t.timepoint == self.timepoint, timepoints)).idx


def construct_capacity_expansion_model(system, model, model_settings):

    N = system.bus
    T = system.tp
    S = system.sc
    L = system.line
    load = system.load

    model.vTHETA = pyo.Var(N, S, T, within=pyo.Reals)

    slack_bus = next(n for n in system.bus if n.bus_type == 'slack')

    Y = build_admittance_matrix2(len(system.bus), system.line)
    B = Y.imag

    model.vTHETA[slack_bus.idx, :, :].fix(0.0)

    model.eFlowAtBus = pyo.Expression(N, S, T, rule=lambda m, n, s, t: 100 * sum(B[n.idx-1, k.idx-1] * (m.vTHETA[n, s, t] - m.vTHETA[k, s, t]) for k in N))
    
    def cMaxFlowPerLine_rule(m, l, s, t):
        if l.rating_MVA <= 0:
            return pyo.Constraint.Skip
        else:
            bus_from = next((n for n in N if n.idx == l.bus_from_idx))
            bus_to = next((n for n in N if n.idx == l.bus_to_idx))
        return (-l.rating_MVA,
                100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * 
                (m.vTHETA[bus_from, s, t] - m.vTHETA[bus_to, s, t]),
                l.rating_MVA)
    
    model.cMaxFlowPerLine = pyo.Constraint(L, S, T, rule=cMaxFlowPerLine_rule)
    
    def cDiffAngle_rule(m, l, s, t):
        if (l.angle_min_deg > -360) and (l.angle_max_deg < 360):
            bus_from = next((n for n in N if n.idx == l.bus_from_idx))
            bus_to = next((n for n in N if n.idx == l.bus_to_idx))
            return (l.angle_min_deg * np.pi / 180, 
                    m.vTHETA[bus_from, s, t] - m.vTHETA[bus_to, s, t],
                    l.angle_max_deg * np.pi / 180)
        else:
            return pyo.Constraint.Skip
        
    model.cDiffAngle = pyo.Constraint(L, S, T, rule=cDiffAngle_rule)

    # Power balance at each bus
    model.cPowerBalance = pyo.Constraint(N, S, T,
                                         rule=lambda m, n, s, t: 
                            m.eGenAtBus[n, s, t] + m.eNetDischargeAtBus[n, s, t] >= 
                            next((ld.load_MW for ld in load 
                                    if  ((ld.bus_idx == n.idx) and 
                                        (ld.scenario_idx == s.idx) and 
                                        (ld.timepoint_idx == t.idx))), 0.0) + m.eFlowAtBus[n, s, t]
                            
                            )

