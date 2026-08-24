import copy
from dataclasses import dataclass

import numpy as np

from sting.utils.dynamical_systems import (
    DynamicalVariables,
    StateSpaceModel,
)


@dataclass(slots=True)
class SeriesRLBranch2A:
    """
    Models a second-order series RL branch in the grid reference frame

    v_from            i ──▶           v_to
     ├─────────VVVVV─────UUUUU─────────┤
                 r         x     
    """

    r_pu: float
    x_pu: float
    wbase: float

    def get_small_signal_model(self,  v_from_D:float, v_from_Q:float, v_to_D: float, v_to_Q:float, i_D:float, i_Q:float):
        """
        d/dt i_DQ = -(r * w_b / x) * i_DQ - j * w * i_DQ + (w_b / x) * v_from_DQ - (w_b / x) * v_to_DQ
        """

        r, x, wb = self.r_pu, self.x_pu, self.wbase

        A = wb * np.array([
            [ -r/x,    1], # Δi_D
            [   -1, -r/x]  # Δi_Q
        ])
        B = np.array([
            #|  Δv_from_DQ  |  Δv_to_DQ 
            [  wb/x,     0, -wb/x,     0], 
            [     0,  wb/x,     0, -wb/x]
        ])

        u = DynamicalVariables(
            name=["v_from_D", "v_from_Q", "v_to_D", "v_to_Q"],
            init=[v_from_D, v_from_Q, v_to_D, v_to_Q],
        )
        x = DynamicalVariables(
            name=["i_br_D", "i_br_Q"], init=[i_D, i_Q])
        y = copy.deepcopy(x)      

        return StateSpaceModel(A=A, B=B, C=np.eye(2), D=np.zeros((2, 4)), u=u, x=x, y=y)


    def get_quadratic_bilinear_model(self, v_from_D:float, v_from_Q:float, v_to_D: float, v_to_Q:float, i_D:float, i_Q:float):
        return self.get_small_signal_model(v_from_D, v_from_Q, v_to_D, v_to_Q, i_D, i_Q).to_quadratic_bilinear()
        