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
class SeriesRLBranch2B:
    """
    Models a second-order series RL branch in an arbitrary reference frame

    v_from            i ──▶           v_to
     ├─────────VVVVV─────UUUUU─────────┤
                 r         x     
    """

    r_pu: float
    x_pu: float
    wbase: float

    def get_small_signal_model(self,  v_from_d:float, v_from_q:float, v_to_d: float, v_to_q:float, i_d:float, i_q:float):
        """
        d/dt i_dq = -(r * w_b / x) * i_dq - j * w * i_dq + (w_b / x) * v_from_dq - (w_b / x) * v_to_dq
        """

        r, x, wb = self.r_pu, self.x_pu, self.wbase

        A = wb * np.array([
            [ -r/x,    1], # Δi_d
            [   -1, -r/x]  # Δi_q
        ])
        B = wb*np.array([
            #|  Δv_from_dq  |  Δv_to_dq  | Δw |
            [  1/x,     0, -1/x,     0,  i_q], 
            [     0,  1/x,     0, -1/x, -i_d]
        ])

        u = DynamicalVariables(
            name=["v_from_d", "v_from_q", "v_to_d", "v_to_q", "w"],
            init=[v_from_d, v_from_q, v_to_d, v_to_q, 1],
        )
        x = DynamicalVariables(
            name=["i_br_d", "i_br_q"], init=[i_d, i_q])
        y = copy.deepcopy(x)      

        return StateSpaceModel(A=A, B=B, C=np.eye(2), D=np.zeros((2, 5)), u=u, x=x, y=y)


    def get_quadratic_bilinear_model(self, v_from_d:float, v_from_q:float, v_to_d: float, v_to_q:float, i_d:float, i_q:float):

        r, x, wb = self.r_pu, self.x_pu, self.wbase

        A = wb * np.array([
            [ -r/x,    0], # i_d
            [    0, -r/x]  # i_q
        ])
        B = wb * np.array([
            #|  v_from_dq  |  v_to_dq  |  w  |
            [  1/x,     0, -1/x,     0,  0], 
            [     0,  1/x,     0, -1/x,  0]
        ])
        N_w = wb * np.array([
            [ 0, 1],# w * i_q
            [-1, 0] # -w * i_d
        ])
        N = np.hstack((np.zeros((2,8)), N_w))

        u = DynamicalVariables(
            name=["v_from_d", "v_from_q", "v_to_d", "v_to_q", "w"],
            init=[v_from_d, v_from_q, v_to_d, v_to_q, 1],
        )
        x = DynamicalVariables(
            name=["i_br_d", "i_br_q"], init=[i_d, i_q])
        y = copy.deepcopy(x)      

        return QuadraticBilinearModel(A=A, B=B, C=np.eye(2), D=np.zeros((2, 5)), H=np.zeros((2,4)), N=N, u=u, x=x, y=y)