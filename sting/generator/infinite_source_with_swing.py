"""
This module implements an infinite source that incorporates:
- A voltage source with variable frequency, swing dynamics, and damping.
- Assumptions at the equilibrium point:
    - The reference angle and the frequency of the source is equal to that of the bus
    - The mechanical power input into the source is equal to the electrical power output from the source
- Series RL branch: It is in series with the voltage source.
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
import polars as pl
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
    p_source: float

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables

# -------------
# Main class
# -------------
@dataclass(slots=True, kw_only=True, eq=False)
class InfiniteSourceWithSwing(Generator):
    r_pu: float
    x_pu: float
    inertia_constant_s: float
    damping_pu: float

    emt_init: InitialConditionsEMT = None

    def _build_small_signal_model(self):

        r = self.r_pu
        x = self.x_pu
        h = self.inertia_constant_s
        d = self.damping_pu

        wb = 2 * np.pi * self.base_frequency_Hz
        angle_ref = self.emt_init.angle_ref * np.pi / 180
        cosphi = np.cos(self.emt_init.angle_ref * np.pi / 180)
        sinphi = np.sin(self.emt_init.angle_ref * np.pi / 180)

        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_int_d, v_int_q = self.emt_init.v_int_d, self.emt_init.v_int_q

        i_bus_D, i_bus_Q = self.emt_init.i_bus_D, self.emt_init.i_bus_Q
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q

        p_source = self.emt_init.p_source

        # Roation matrix (turn off code formatters for matrices)
        # fmt: off
        R = np.array(
            [[cosphi, -sinphi], 
             [sinphi, cosphi]])

        # Define state-space matrices 
        A = wb * np.array(
            [[0,                                            1/wb,                   0,              0               ],
             [0,                                            -d / (2.0 * h * wb),    -v_int_d/(2*h), -v_int_q/(2*h)  ],
             [1/x * (sinphi * v_bus_D - cosphi * v_bus_Q),  i_bus_q/wb,             -r/x,           1               ],
             [1/x * (cosphi * v_bus_D + sinphi * v_bus_Q),  -i_bus_d/wb,            -1,             -r/x            ]])
        
        B = wb * np.array(
            [[0,        0,              0,              0,          0           ],  
             [1/(2*h),  -i_bus_d/(2*h), -i_bus_q/(2*h), 0,          0           ],
             [0,        1/x,            0,              -cosphi/x,  -sinphi/x   ],
             [0,        0,              1/x,            sinphi/x,   -cosphi/x    ]]) 
        # B = B @ block_diag(np.eye(2), R.T) 
        # fmt: on
        C = np.hstack((np.zeros((2, 2)), R))

        D = np.zeros((2, 5))

        # Inputs
        u = DynamicalVariables(
            name=["p_m", "v_ref_d", "v_ref_q", "v_bus_D", "v_bus_Q"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "device", "grid", "grid"],
            init=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            component=f"{self.type_}_{self.id}",
            init=[0.0, 0.0],
        )

        # States
        x = DynamicalVariables(
            name=["delta", "omega", "i_bus_d", "i_bus_q"],
            component=f"{self.type_}_{self.id}",
            init=[0.0, 0.0, 0.0, 0.0],
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

        p_source = v_int_dq * i_bus_dq.conjugate()

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
            p_source=p_source.real,
        )