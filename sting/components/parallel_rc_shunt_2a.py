import copy
from dataclasses import dataclass

import numpy as np

from sting.utils.dynamical_systems import (
    DynamicalVariables,
    StateSpaceModel,
)


@dataclass(slots=True)
class ParallelRCShunt2A:
    """
    Models a second-order series RC shunt in the grid reference frame

    i_DQ │  ──┬── v_DQ
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

    def get_small_signal_model(self, v_D:float, v_Q:float, i_D: float, i_Q:float):
        """
        d/dt v_DQ = -(g * w_b / b) * v_DQ - j * w_b * v_DQ + (w_b / b) * i_DQ
        """

        g, b, wb = self.g_pu, self.b_pu, self.wbase

        A = wb * np.array([
            [ -g/b,    1], # Δv_D
            [   -1, -g/b]  # Δv_Q
        ])
        B = np.array([
            [ wb/b,     0], 
            [     0, wb/b]
        ])

        u = DynamicalVariables(name=["i_sh_D", "i_sh_Q"], init=[i_D, i_Q])
        x = DynamicalVariables(name=["v_sh_D", "v_sh_Q"], init=[v_D, v_Q])
        y = copy.deepcopy(x)      

        return StateSpaceModel(A=A, B=B, C=np.eye(2), D=np.zeros((2, 2)), u=u, x=x, y=y)


    def get_quadratic_bilinear_model(self, v_D:float, v_Q:float, i_D: float, i_Q:float):
        return self.get_small_signal_model( v_D, v_Q, i_D, i_Q).to_quadratic_bilinear()