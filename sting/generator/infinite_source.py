"""
This module implements an infinite source that incorporates:
- Stiff voltage source: a voltage source with constant frequency and constant voltage.
- Series RL branch: It is in series with the stiff voltage source.
"""

import numpy as np
from scipy.linalg import block_diag
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, ClassVar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.utils.transformations import dq02abc, abc2dq0

class PowerFlowVariables(NamedTuple):
    p_bus: float
    q_bus: float
    vmag_bus: float
    vphase_bus: float


class InitialConditionsEMT(NamedTuple):
    v_bus_D: float
    v_bus_Q: float
    v_int_d: float
    v_int_q: float
    i_bus_d: float
    i_bus_q: float
    i_bus_D: float
    i_bus_Q: float
    angle_ref: float

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables


@dataclass(slots=True)
class InfiniteSource:
    id: int = field(default=0, init=False)
    name: str 
    bus: str
    p_min: float
    p_max: float
    q_min: float
    q_max: float
    sbase: float
    vbase: float
    fbase: float
    r: float
    l: float
    bus_id: int = None
    pf: Optional[PowerFlowVariables] = None
    emt_init: Optional[InitialConditionsEMT] = None
    ssm: Optional[StateSpaceModel] = None
    type: str = "inf_src"
    tags: ClassVar[list[str]] = ["generator"]
    variables_emt: Optional[VariablesEMT] = None
    id_variables_emt: Optional[dict] = None

    def assign_indices(self, system):
        self.bus_id = next((n for n in system.bus if n.name == self.bus)).id

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.generators.loc[f"{self.type}_{self.id}"]
        self.pf = PowerFlowVariables(
            p_bus=sol.p.item(),
            q_bus=sol.q.item(),
            vmag_bus=sol.bus_vmag.item(),
            vphase_bus=sol.bus_vphase.item(),
        )

    def _build_small_signal_model(self):

        r = self.r
        l = self.l

        wb = 2 * np.pi * self.fbase
        cosphi = np.cos(self.emt_init.angle_ref * np.pi / 180)
        sinphi = np.sin(self.emt_init.angle_ref * np.pi / 180)

        # Roation matrix (turn off code formatters for matrices)
        # fmt: off
        R = np.array(
            [[cosphi, -sinphi], 
             [sinphi, cosphi]])

        # Define state-space matrices 
        A = wb * np.array(
            [[-r/l,    1], 
             [  -1, -r/l]])

        B = wb * np.array(
            [[  1/l,    0, -1/l,    0], 
             [    0,  1/l,    0, -1/l]]) 
        B = B @ block_diag(np.eye(2), R.T) 
        # fmt: on
        C = R

        D = np.zeros((2, 4))

        # Inputs
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_int_d, v_int_q = self.emt_init.v_int_d, self.emt_init.v_int_q

        u = DynamicalVariables(
            name=["v_bus_D", "v_bus_Q", "v_ref_d", "v_ref_q"],
            component=f"{self.type}_{self.id}",
            type=["grid", "grid", "device", "device"],
            init=[v_bus_D, v_bus_Q, v_int_d, v_int_q],
        )

        # Outputs
        i_bus_D, i_bus_Q = self.emt_init.i_bus_D, self.emt_init.i_bus_Q

        y = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            component=f"{self.type}_{self.id}",
            init=[i_bus_D, i_bus_Q],
        )

        # States
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q

        x = DynamicalVariables(
            name=["i_bus_d", "i_bus_q"],
            component=f"{self.type}_{self.id}",
            init=[i_bus_d, i_bus_q],
        )

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)

    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.pf.vmag_bus
        vphase_bus = self.pf.vphase_bus
        p_bus = self.pf.p_bus
        q_bus = self.pf.q_bus

        v_bus_DQ = vmag_bus * np.exp(vphase_bus * 1j * np.pi / 180)
        i_bus_DQ = ((p_bus + 1j * q_bus) / v_bus_DQ).conjugate()

        v_int_DQ = v_bus_DQ + i_bus_DQ * (self.r + 1j * self.l)
        angle_ref = np.angle(v_int_DQ, deg=True)

        v_int_dq = v_int_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_bus_dq = i_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        self.emt_init = InitialConditionsEMT(
            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag,
            v_int_d=v_int_dq.real,
            v_int_q=v_int_dq.imag,
            i_bus_d=i_bus_dq.real,
            i_bus_q=i_bus_dq.imag,
            i_bus_D=i_bus_DQ.real,
            i_bus_Q=i_bus_DQ.imag,
            angle_ref=angle_ref,
        )

    def define_variables_emt(self):
        
        # States
        # ------

        # Initial conditions
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q
        angle_ref = self.emt_init.angle_ref * np.pi / 180
        i_bus_a, i_bus_b, i_bus_c = dq02abc(i_bus_d, i_bus_q, 0, angle_ref)

        
        x = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c", "angle_ref"],
            component=f"{self.type}_{self.id}",
            init=[i_bus_a, i_bus_b, i_bus_c, angle_ref],
        )

        # Inputs
        # ------

        # Initial conditions
        v_ref_d, v_ref_q = self.emt_init.v_int_d, self.emt_init.v_int_q
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, 0)

        u = DynamicalVariables(
            name=["v_ref_d", "v_ref_q", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type}_{self.id}",
            type=["device", "device", "grid", "grid", "grid"],
            init=[v_ref_d, v_ref_q, v_bus_a, v_bus_b, v_bus_c],
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type}_{self.id}",
        )

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)
    
    def get_derivative_state_emt(self):

        # Get state values
        i_bus_a, i_bus_b, i_bus_c, angle_ref = self.variables_emt.x.value

        # Get input values
        v_ref_d, v_ref_q, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        v_ref_a, v_ref_b, v_ref_c = dq02abc(v_ref_d, v_ref_q, 0, angle_ref)

        # Get parameters
        r = self.r
        l = self.l
        wb = 2 * np.pi * self.fbase

        # Differential equations
        d_i_bus_a = wb / l * (v_ref_a - v_bus_a - r * i_bus_a)
        d_i_bus_b = wb / l * (v_ref_b - v_bus_b - r * i_bus_b)
        d_i_bus_c = wb / l * (v_ref_c - v_bus_c - r * i_bus_c)
        d_angle_ref = wb 

        return [d_i_bus_a, d_i_bus_b, d_i_bus_c, d_angle_ref]
    
    def get_output_emt(self):
        
        i_bus_a, i_bus_b, i_bus_c, angle_ref = self.variables_emt.x.value

        return [i_bus_a, i_bus_b, i_bus_c]
    
    def plot_results_emt(self, output_dir):
        """
        Plot EMT simulation results
        """

        i_bus_a, i_bus_b, i_bus_c, angle_ref = self.variables_emt.x.value
        i_bus_d, i_bus_q, _ = zip(*map(abc2dq0, i_bus_a, i_bus_b, i_bus_c, angle_ref))
        t = self.variables_emt.x.time

        fig = make_subplots(rows=1, cols=2)
        
        fig.add_trace(go.Scatter(x=t, y=i_bus_d), row=1, col=1)
        fig.update_xaxes(title_text='Time [s]', row=1, col=1)
        fig.update_yaxes(title_text='i_bus_d [p.u.]', row=1, col=1)

        fig.add_trace(go.Scatter(x=t, y=i_bus_q), row=1, col=2)
        fig.update_xaxes(title_text='Time [s]', row=1, col=2)
        fig.update_yaxes(title_text='i_bus_q [p.u.]', row=1, col=2)

        name = f"{self.type}_{self.idx}"
        fig.update_layout(  title_text = name,
                            title_x=0.5,
                            showlegend = False,
                            )

        fig.write_html(os.path.join(output_dir, name + ".html"))