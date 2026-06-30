"""
This module implements a voltage source that incorporates:
- A variable frequency and swing dynamics.
- Equilibrium assumptions:
    - The source frequency equals to the nominal bus frequency.
    - Mechanical input power equals to the electrical output power of the source.
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
class SourceWithSwing(Generator):
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
        C = np.hstack((np.array([[-i_bus_Q, 0], [i_bus_D, 0]]), R))

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

    def define_variables_emt(self):
        
        # States
        # ------

        # Initial conditions
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q
        angle_ref = self.emt_init.angle_ref * np.pi / 180
        i_bus_a, i_bus_b, i_bus_c = dq02abc(i_bus_d, i_bus_q, 0, angle_ref)

        wb = 2 * np.pi * self.base_frequency_Hz

        x = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c", "angle_ref", "omega"],
            component=f"{self.type_}_{self.id}",
            init=[i_bus_a, i_bus_b, i_bus_c, angle_ref, wb],
        )

        # Inputs
        # ------

        # Initial conditions
        v_ref_d, v_ref_q = self.emt_init.v_int_d, self.emt_init.v_int_q
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, 0)
        p_m = self.emt_init.p_source

        u = DynamicalVariables(
            name=["p_m", "v_ref_d", "v_ref_q", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "device", "grid", "grid", "grid"],
            init=[p_m, v_ref_d, v_ref_q, v_bus_a, v_bus_b, v_bus_c],
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
        )

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)

    def get_derivative_state_emt(self):

        # Get state values
        i_bus_a, i_bus_b, i_bus_c, angle_ref, omega = self.variables_emt.x.value

        # Get input values
        p_m, v_ref_d, v_ref_q, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        v_ref_a, v_ref_b, v_ref_c = dq02abc(v_ref_d, v_ref_q, 0, angle_ref)
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, angle_ref)

        # Get parameters
        r = self.r_pu
        x = self.x_pu
        h = self.inertia_constant_s
        d = self.damping_pu

        wb = 2 * np.pi * self.base_frequency_Hz

        p_e = v_ref_d * i_bus_d + v_ref_q * i_bus_q

        # Differential equations
        d_i_bus_a = wb / x * (v_ref_a - v_bus_a - r * i_bus_a)
        d_i_bus_b = wb / x * (v_ref_b - v_bus_b - r * i_bus_b)
        d_i_bus_c = wb / x * (v_ref_c - v_bus_c - r * i_bus_c)
        d_angle_ref = omega 
        d_omega = wb / (2.0 * h) * (p_m - p_e - d * (omega - wb) / wb)

        return [d_i_bus_a, d_i_bus_b, d_i_bus_c, d_angle_ref, d_omega]
    
    def get_output_emt(self):

        i_bus_a, i_bus_b, i_bus_c, angle_ref, omega = self.variables_emt.x.value

        return [i_bus_a, i_bus_b, i_bus_c]
    
    def plot_results_emt(self):
        """
        Plot EMT simulation results
        """

        i_bus_a, i_bus_b, i_bus_c, angle_ref, omega = self.variables_emt.x.value
        i_bus_d, i_bus_q, _ = zip(*map(abc2dq0, i_bus_a, i_bus_b, i_bus_c, angle_ref))
        t = self.variables_emt.x.time

        wb = 2.0 * np.pi * self.base_frequency_Hz
        angle_ref_init = self.emt_init.angle_ref * np.pi / 180.0

        delta_dev = angle_ref - angle_ref_init - wb * t
        omega_dev = omega - wb
        i_bus_d_dev = i_bus_d - self.emt_init.i_bus_d
        i_bus_q_dev = i_bus_q - self.emt_init.i_bus_q

        results = DynamicalVariables(
            name=["delta", "omega", "i_bus_d", "i_bus_q"],
            component=f"{self.type_}_{self.id}",
            value=[delta_dev, omega_dev, i_bus_d_dev, i_bus_q_dev],
            time=t,
        )
        return results


    def compare_ssm_emt(self, emt_directory, ssm_directory):
        # Read the SSM and EMT states
        emt = pl.read_csv(os.path.join(emt_directory, f"{self.type_}_{self.id}_states.csv"))
        ssm = pl.read_csv(os.path.join(ssm_directory, f"{self.type_}_{self.id}_states.csv"))

        # Transform EMT abc states to dq0 states
        
        i_a, i_b, i_c, angle_ref, omega = [c.to_numpy() for c in emt.select("i_bus_a", "i_bus_b", "i_bus_c", "angle_ref", "omega")]
        i_emt_d, i_emt_q, _ = zip(*map(abc2dq0, i_a, i_b, i_c, angle_ref))
        t = emt["time"].to_numpy()
        
        wb = 2.0 * np.pi * self.base_frequency_Hz
        angle_ref_init = self.emt_init.angle_ref * np.pi / 180.0

        delta_emt_dev = (angle_ref - angle_ref_init - wb * t)
        omega_emt_dev = omega - wb
        i_emt_d_dev = i_emt_d - self.emt_init.i_bus_d
        i_emt_q_dev = i_emt_q - self.emt_init.i_bus_q

        # Unpack the SSM dq states
        delta_ssm, omega_ssm, i_ssm_d, i_ssm_q = [c.to_numpy() for c in ssm.select("delta", "omega", "i_bus_d", "i_bus_q")]

        # Return deltas
        return {
            f"({self.type_}_{self.id}, delta)": (delta_emt_dev, delta_ssm),
            f"({self.type_}_{self.id}, omega)": (omega_emt_dev, omega_ssm),
            f"({self.type_}_{self.id}, i_bus_d)": (i_emt_d_dev, i_ssm_d),
            f"({self.type_}_{self.id}, i_bus_q)": (i_emt_q_dev, i_ssm_q)
        }