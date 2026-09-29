import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.shunt.core import Shunt, VariablesEMT
from sting.utils.dynamical_systems import (
    DynamicalVariables,
    QuadraticBilinearModel,
    StateSpaceModel,
)
from sting.utils.transformations import abc2dq0, dq02abc


class InitialConditionsEMT(NamedTuple):
    vmag_bus: float
    vphase_bus: float
    v_bus_D: float
    v_bus_Q: float
    v_bus_a: float
    v_bus_b: float
    v_bus_c: float
    i_bus_D: float
    i_bus_Q: float


@dataclass(slots=True)
class ParallelRCShunt2A(Shunt):
    """
    Models a second-order series RC shunt in an arbitrary reference frame

    i_dq │  ──┬── v_dq
         ▼    │
          ┌───┴───┐
     g_pu <      ─┴─ b_pu
          >      ─┬─
          └───┬───┘
              │
           Neutral
    """

    g_pu: float # conductance
    b_pu: float # susceptance

    emt_init: InitialConditionsEMT = None

    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz

    def get_steady_state(self, v_sh_mag, v_sh_phase):
        g = self.g_pu
        b = self.b_pu

        v_bus_DQ = v_sh_mag * np.exp(v_sh_phase * 1j * np.pi / 180)
        i_bus_DQ = v_bus_DQ * g + v_bus_DQ * (1j * b)
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_DQ.real, v_bus_DQ.imag, 0, 0)

        self.emt_init = InitialConditionsEMT(
            vmag_bus=v_sh_mag,
            vphase_bus=v_sh_phase,

            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag,
            v_bus_a=v_bus_a, 
            v_bus_b=v_bus_b, 
            v_bus_c=v_bus_c,

            i_bus_D=i_bus_DQ.real,
            i_bus_Q=i_bus_DQ.imag,
        )


    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.power_flow_variables.vmag_bus
        vphase_bus = self.power_flow_variables.vphase_bus
        self.get_steady_state(vmag_bus, vphase_bus)


    def get_small_signal_model(self, v_d, v_q, i_d, i_q):
        """
        d/dt v_dq = -(g * w_b / b) * v_dq - j * w * v_dq + (w_b / b) * i_dq
        """
        g, b, wb = self.g_pu, self.b_pu, self.wbase

        A = wb * np.array([
            [ -g/b,    1], # Δv_d
            [   -1, -g/b]  # Δv_q
        ])
        B = wb*np.array([
            [+v_q, 1/b,   0], 
            [-v_d,   0, 1/b]
        ])

        u = DynamicalVariables(
            name=["w_slack", "i_sh_d", "i_sh_q"], 
            init=[1, i_d, i_q],
            component=f"{self.type_}_{self.id}",
            type=["device", "grid", "grid"],
            )
        x = DynamicalVariables(name=["v_sh_d", "v_sh_q"], init=[v_d, v_q], component=f"{self.type_}_{self.id}")
        y = copy.deepcopy(x)      

        return StateSpaceModel(A=A, B=B, C=np.eye(2), D=np.zeros((2, 3)), u=u, x=x, y=y)

    def _build_small_signal_model(self):
        i_d, i_q = self.emt_init.i_bus_D, self.emt_init.i_bus_Q
        v_d, v_q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        self.ssm =self.get_small_signal_model(v_d, v_q, i_d, i_q)

    def get_quadratic_bilinear_model(self, v_d, v_q, i_d, i_q):
        g, b, wb = self.g_pu, self.b_pu, self.wbase
        A = wb * np.array([
            [-g/b,    0], # Δv_d
            [   0, -g/b]  # Δv_q
        ])
        B = np.array([
            [0, wb/b,     0], 
            [0,     0, wb/b]
        ])
        N_w = wb* np.array([
            [ 0, 1], # w * v_q
            [-1, 0]  # -w * v_d
        ])
        N = np.hstack([N_w, np.zeros((2,4))])

        u = DynamicalVariables(
            name=["w_slack", "i_sh_d", "i_sh_q"], 
            init=[1, i_d, i_q],
            component=f"{self.type_}_{self.id}",
            type=["device", "grid", "grid"],
            )
        x = DynamicalVariables(name=["v_sh_d", "v_sh_q"], init=[v_d, v_q], component=f"{self.type_}_{self.id}")
        y = copy.deepcopy(x)
        qbm = QuadraticBilinearModel(A=A, B=B, C=np.eye(2), D=np.zeros((2, 3)), H=np.zeros((2,4)), N=N, u=u, x=x, y=y)
        return qbm

    def _build_quadratic_bilinear_model(self):
        i_d, i_q = self.emt_init.i_bus_D, self.emt_init.i_bus_Q
        v_d, v_q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        self.qbm = self.get_quadratic_bilinear_model(v_d, v_q, i_d, i_q)

    def define_variables_emt(self):

        # States
        # ------
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, 0)

        x = DynamicalVariables(
            name=["v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[v_bus_a, v_bus_b, v_bus_c],
        )

        # Inputs
        u = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["grid", "grid", "grid"],
        )

        # Outputs
        y = DynamicalVariables(
            name=["v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
        )

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)

    def get_derivatives_step_emt_abc(self, v_sh_a, v_sh_b, v_sh_c, i_sh_a, i_sh_b, i_sh_c):
        return self.get_derivative_state_emt([v_sh_a, v_sh_b, v_sh_c], [i_sh_a, i_sh_b, i_sh_c])

    def get_derivative_state_emt(self, x, u):

        # Get state values
        v_bus_a, v_bus_b, v_bus_c = x

        # Get input values
        i_bus_a, i_bus_b, i_bus_c = u

        # Get parameters
        g = self.g_pu
        b = self.b_pu
        wb = 2 * np.pi * self.base_frequency_Hz

        # Differential equations
        d_v_bus_a = wb / b * (- g * v_bus_a + i_bus_a)
        d_v_bus_b = wb / b * (- g * v_bus_b + i_bus_b)
        d_v_bus_c = wb / b * (- g * v_bus_c + i_bus_c)

        return [d_v_bus_a, d_v_bus_b, d_v_bus_c]
    
    def get_output_emt(self, x):

        v_bus_a, v_bus_b, v_bus_c = x

        return [v_bus_a, v_bus_b, v_bus_c]
    
    def plot_results_emt(self):

        # Get state values
        v_bus_a, v_bus_b, v_bus_c = self.variables_emt.x.value
        time = self.variables_emt.x.time
        # Assumes a reference frame rotating at w_base
        angle_ref =  2 * np.pi * self.base_frequency_Hz * time

        # Transform abc to dq0
        v_bus_D, v_bus_Q, _ = zip(*map(abc2dq0, v_bus_a, v_bus_b, v_bus_c, angle_ref))
        
        results = DynamicalVariables(
            name=["v_bus_D", "v_bus_Q"],
            component=f"{self.type_}_{self.id}",
            value=[v_bus_D, v_bus_Q],
            time=time
        )
        return results