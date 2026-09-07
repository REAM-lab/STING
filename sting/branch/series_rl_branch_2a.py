import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.branch.core import Branch, VariablesEMT
from sting.utils.dynamical_systems import (
    DynamicalVariables,
    QuadraticBilinearModel,
    StateSpaceModel,
)
from sting.utils.transformations import abc2dq0, dq02abc


class InitialConditionsEMT(NamedTuple):
    vmag_from_bus: float
    vphase_from_bus: float
    vmag_to_bus: float
    vphase_to_bus: float
    v_from_bus_D: float
    v_from_bus_Q: float
    v_to_bus_D: float
    v_to_bus_Q: float
    i_br_D: float
    i_br_Q: float


@dataclass(slots=True)
class SeriesRLBranch2A(Branch):
    """
    Models a second-order series RL branch in an arbitrary reference frame

    v_from            i ──▶           v_to
     ├─────────VVVVV─────UUUUU─────────┤
                 r         x
    """

    r_pu: float
    x_pu: float

    emt_init: InitialConditionsEMT = None

    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz

    def _calculate_emt_initial_conditions(self):
        r = self.r_pu
        x = self.x_pu

        vmag_from_bus = self.power_flow_variables.vmag_from_bus
        vphase_from_bus = self.power_flow_variables.vphase_from_bus

        vmag_to_bus = self.power_flow_variables.vmag_to_bus
        vphase_to_bus = self.power_flow_variables.vphase_to_bus

        v_from_bus_DQ = vmag_from_bus * np.exp(vphase_from_bus * np.pi / 180 * 1j)
        v_to_bus_DQ = vmag_to_bus * np.exp(vphase_to_bus * np.pi / 180 * 1j)

        i_br_DQ = (v_from_bus_DQ - v_to_bus_DQ) / (r + 1j * x)

        self.emt_init = InitialConditionsEMT(
            vmag_from_bus=vmag_from_bus,
            vphase_from_bus=vphase_from_bus,
            vmag_to_bus=vmag_to_bus,
            vphase_to_bus=vphase_to_bus,
            v_from_bus_D=v_from_bus_DQ.real,
            v_from_bus_Q=v_from_bus_DQ.imag,
            v_to_bus_D=v_to_bus_DQ.real,
            v_to_bus_Q=v_to_bus_DQ.imag,
            i_br_D=i_br_DQ.real,
            i_br_Q=i_br_DQ.imag,
        )

    def _build_small_signal_model(self):
        """
        d/dt i_dq = -(r * w_b / x) * i_dq - j * w * i_dq + (w_b / x) * v_from_dq - (w_b / x) * v_to_dq
        """
        # Parameters
        r, x, wb = self.r_pu, self.x_pu, self.wbase
        # Initial conditions
        i_d, i_q = self.emt_init.i_br_D, self.emt_init.i_br_Q
        v_from_d, v_from_q = self.emt_init.v_from_bus_D, self.emt_init.v_from_bus_Q
        v_to_d, v_to_q = self.emt_init.v_to_bus_D, self.emt_init.v_to_bus_Q

        A = wb * np.array([
                [-r/x, 1   ],  # Δi_d
                [  -1, -r/x],  # Δi_q
            ])
        B = wb * np.array([
                [+i_q, 1/x,   0,-1/x,   0],
                [-i_d,   0, 1/x,   0,-1/x],
        ])

        u = DynamicalVariables(
            name=["w_slack", "v_from_d", "v_from_q", "v_to_d", "v_to_q"],
            init=[1, v_from_d, v_from_q, v_to_d, v_to_q],
            component=f"{self.type_}_{self.id}",
            type=["device", "grid", "grid", "grid", "grid"],
        )
        x = DynamicalVariables(name=["i_br_d", "i_br_q"], init=[i_d, i_q], component=f"{self.type_}_{self.id}")
        y = copy.deepcopy(x)

        self.ssm = StateSpaceModel(A=A, B=B, C=np.eye(2), D=np.zeros((2, 5)), u=u, x=x, y=y)
        return self.ssm

    def _build_quadratic_bilinear_model(self):
        # Parameters
        r, x, wb = self.r_pu, self.x_pu, self.wbase
        # Initial conditions
        i_d, i_q = self.emt_init.i_br_D, self.emt_init.i_br_Q
        v_from_d, v_from_q = self.emt_init.v_from_bus_D, self.emt_init.v_from_bus_Q
        v_to_d, v_to_q = self.emt_init.v_to_bus_D, self.emt_init.v_to_bus_Q

        A = (wb/x) * np.array([
                [-r, 0],  # i_d
                [0, -r],  # i_q
            ])
        B = (wb/x) * np.array([
                [0, 1, 0, -1, 0],
                [0, 0, 1, 0, -1],
            ])
        N_w = wb * np.array([
                [ 0, 1],  # w * i_q
                [-1, 0],  # -w * i_d
            ])
        N = np.hstack((N_w, np.zeros((2, 8))))

        u = DynamicalVariables(
            name=["w_slack", "v_from_d", "v_from_q", "v_to_d", "v_to_q"],
            init=[1, v_from_d, v_from_q, v_to_d, v_to_q],
            component=f"{self.type_}_{self.id}",
            type=["device", "grid", "grid", "grid", "grid"],
        )
        x = DynamicalVariables(name=["i_br_d", "i_br_q"], init=[i_d, i_q], component=f"{self.type_}_{self.id}")
        y = copy.deepcopy(x)

        self.qbm = QuadraticBilinearModel(
            A=A,
            B=B,
            C=np.eye(2),
            D=np.zeros((2, 5)),
            H=np.zeros((2, 4)),
            N=N,
            u=u,
            x=x,
            y=y,
        )
        return self.qbm

    def define_variables_emt(self):

        # States
        # ------
        i_br_D, i_br_Q = self.emt_init.i_br_D, self.emt_init.i_br_Q
        i_br_a, i_br_b, i_br_c = dq02abc(i_br_D, i_br_Q, 0, 0)

        x = DynamicalVariables(
            name=["i_br_a", "i_br_b", "i_br_c"],
            component=f"{self.type_}_{self.id}",
            init=[i_br_a, i_br_b, i_br_c],
        )

        # Inputs
        u = DynamicalVariables(
            name=[
                "v_from_bus_a",
                "v_from_bus_b",
                "v_from_bus_c",
                "v_to_bus_a",
                "v_to_bus_b",
                "v_to_bus_c",
            ],
            component=f"{self.type_}_{self.id}",
            type=["grid", "grid", "grid", "grid", "grid", "grid"],
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_br_a", "i_br_b", "i_br_c"],
            component=f"{self.type_}_{self.id}",
        )

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)

    def get_derivative_state_emt(self, x, u):

        # Get state values
        i_br_a, i_br_b, i_br_c = x

        # Get input values
        v_from_bus_a, v_from_bus_b, v_from_bus_c, v_to_bus_a, v_to_bus_b, v_to_bus_c = u

        # Get parameters
        r = self.r_pu
        xf = self.x_pu
        wb = 2 * np.pi * self.base_frequency_Hz

        # Differential equations
        d_i_br_a = wb / xf * (v_from_bus_a - v_to_bus_a - r * i_br_a)
        d_i_br_b = wb / xf * (v_from_bus_b - v_to_bus_b - r * i_br_b)
        d_i_br_c = wb / xf * (v_from_bus_c - v_to_bus_c - r * i_br_c)

        return [d_i_br_a, d_i_br_b, d_i_br_c]

    def get_output_emt(self, x):

        i_br_a, i_br_b, i_br_c = x

        return [i_br_a, i_br_b, i_br_c]

    def plot_results_emt(self):

        # Retrieve simulation results
        time = self.variables_emt.x.time
        # Assumes a reference frame rotating at w_base
        angle_ref = 2 * np.pi * self.base_frequency_Hz * time
        i_br_a, i_br_b, i_br_c = self.variables_emt.x.value

        # Transform abc to dq0
        i_br_D, i_br_Q, _ = zip(*map(abc2dq0, i_br_a, i_br_b, i_br_c, angle_ref))

        # Plot results
        results = DynamicalVariables(
            name=["i_br_D", "i_br_Q"],
            component=f"{self.type_}_{self.id}",
            value=[i_br_D, i_br_Q],
            time=time,
        )
        return results
