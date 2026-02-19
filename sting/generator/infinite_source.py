"""
This module implements an infinite source that incorporates:
- Stiff voltage source: a voltage source with constant frequency and constant voltage.
- Series RL branch: It is in series with the stiff voltage source.
"""
# -------------
# Import python packages
# --------------
import numpy as np
from scipy.linalg import block_diag
from dataclasses import dataclass
from typing import NamedTuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# -------------
# Import sting code
# -------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.utils.transformations import dq02abc, abc2dq0
from sting.generator.core import Generator

# -------------
# Sub-classes
# -------------
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


@dataclass(slots=True, kw_only=True, eq=False)
class InfiniteSource(Generator):
    r_pu: float
    x_pu: float
    emt_init: InitialConditionsEMT = None
    ssm: StateSpaceModel = None
    variables_emt: VariablesEMT = None
    id_variables_emt: dict = None

    def _build_small_signal_model(self):

        r = self.r_pu
        x = self.x_pu

        wb = 2 * np.pi * self.base_frequency_Hz
        cosphi = np.cos(self.emt_init.angle_ref * np.pi / 180)
        sinphi = np.sin(self.emt_init.angle_ref * np.pi / 180)

        # Roation matrix (turn off code formatters for matrices)
        # fmt: off
        R = np.array(
            [[cosphi, -sinphi], 
             [sinphi, cosphi]])

        # Define state-space matrices 
        A = wb * np.array(
            [[-r/x,    1], 
             [  -1, -r/x]])

        B = wb * np.array(
            [[  1/x,    0, -1/x,    0], 
             [    0,  1/x,    0, -1/x]]) 
        B = B @ block_diag(np.eye(2), R.T) 
        # fmt: on
        C = R

        D = np.zeros((2, 4))

        # Inputs
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_int_d, v_int_q = self.emt_init.v_int_d, self.emt_init.v_int_q

        u = DynamicalVariables(
            name=["v_ref_d", "v_ref_q", "v_bus_D", "v_bus_Q"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "grid", "grid"],
            init=[v_int_d, v_int_q, v_bus_D, v_bus_Q],
        )

        # Outputs
        i_bus_D, i_bus_Q = self.emt_init.i_bus_D, self.emt_init.i_bus_Q

        y = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            component=f"{self.type_}_{self.id}",
            init=[i_bus_D, i_bus_Q],
        )

        # States
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q

        x = DynamicalVariables(
            name=["i_bus_d", "i_bus_q"],
            component=f"{self.type_}_{self.id}",
            init=[i_bus_d, i_bus_q],
        )

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)

    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.power_flow_variables.vmag_bus
        vphase_bus = self.power_flow_variables.vphase_bus
        p_bus = self.power_flow_variables.p_bus
        q_bus = self.power_flow_variables.q_bus

        v_bus_DQ = vmag_bus * np.exp(vphase_bus * 1j * np.pi / 180)
        i_bus_DQ = ((p_bus + 1j * q_bus) / v_bus_DQ).conjugate()

        v_int_DQ = v_bus_DQ + i_bus_DQ * (self.r_pu + 1j * self.x_pu)
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
            component=f"{self.type_}_{self.id}",
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
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "grid", "grid", "grid"],
            init=[v_ref_d, v_ref_q, v_bus_a, v_bus_b, v_bus_c],
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
        )

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)
    
    def get_derivative_state_emt(self):

        # Get state values
        i_bus_a, i_bus_b, i_bus_c, angle_ref = self.variables_emt.x.value

        # Get input values
        v_ref_d, v_ref_q, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        v_ref_a, v_ref_b, v_ref_c = dq02abc(v_ref_d, v_ref_q, 0, angle_ref)

        # Get parameters
        r = self.r_pu
        x = self.x_pu
        wb = 2 * np.pi * self.base_frequency_Hz

        # Differential equations
        d_i_bus_a = wb / x * (v_ref_a - v_bus_a - r * i_bus_a)
        d_i_bus_b = wb / x * (v_ref_b - v_bus_b - r * i_bus_b)
        d_i_bus_c = wb / x * (v_ref_c - v_bus_c - r * i_bus_c)
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

        name = f"{self.type_}_{self.id}"
        fig.update_layout(  title_text = name,
                            title_x=0.5,
                            showlegend = False,
                            )

        fig.write_html(os.path.join(output_dir, name + ".html"))