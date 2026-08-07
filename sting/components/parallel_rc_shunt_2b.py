import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.utils.dynamical_systems import (
    DynamicalVariables,
    QuadraticBilinearModel,
    StateSpaceModel,
)

@dataclass(slots=True)
class ParallelRCShunt2B:
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
    wbase: float

    def get_steady_state(self, v_d:float, v_q:float, i_d: float, i_q:float):
        pass

    def get_small_signal_model(self, v_d:float, v_q:float, i_d: float, i_q:float):
        """
        d/dt v_dq = -(g * w_b / b) * v_dq - j * w * v_dq + (w_b / b) * i_dq
        """

        g, b, wb = self.g_pu, self.b_pu, self.wbase

        A = wb * np.array([
            [ -g/b,    1], # Δv_d
            [   -1, -g/b]  # Δv_q
        ])
        B = np.array([
            #|   Δi_dq   | Δw |
            [ wb/b,     0,  v_q], 
            [     0, wb/b, -v_d]
        ])

        u = DynamicalVariables(name=["i_sh_d", "i_sh_q", "w"], init=[i_d, i_q, wb])
        x = DynamicalVariables(name=["v_sh_d", "v_sh_q"], init=[v_d, v_q])
        y = copy.deepcopy(x)      

        return StateSpaceModel(A=A, B=B, C=np.eye(2), D=np.zeros((2, 5)), u=u, x=x, y=y)


    def get_quadratic_bilinear_model(self, v_d:float, v_q:float, i_d: float, i_q:float):

        g, b, wb = self.g_pu, self.b_pu, self.wbase

        A = wb * np.array([
            [ -g/b,    1], # Δv_d
            [   -1, -g/b]  # Δv_q
        ])
        B = np.array([
            #|   Δi_dq   | Δw |
            [ wb/b,     0,  0], 
            [     0, wb/b,  0]
        ])
        N_w = np.array([
            [ 0, 1], # w * v_q
            [-1, 0]  # -w * v_d
        ])
        N = np.hstack(np.zeros(2,8), N_w)

        u = DynamicalVariables(name=["i_sh_d", "i_sh_q", "w"], init=[i_d, i_q, wb])
        x = DynamicalVariables(name=["v_sh_d", "v_sh_q"], init=[v_d, v_q])
        y = copy.deepcopy(x)          

        return QuadraticBilinearModel(A=A, B=B, C=np.eye(2), D=np.zeros((2, 5)), H=np.zeros((2,4)), N=N, u=u, x=x, y=y)


    def define_variables_emt_abc(self):
        pass


    def get_derivatives_step_emt_abc(self):
        pass


    def get_algebraics_step_emt_abc(self):
        pass