import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.utils.dynamical_systems import (
    DynamicalVariables,
    StateSpaceModel,
)
from sting.utils.transformations import abc2dq0, dq02abc

class InitialConditionsEMT(NamedTuple):
    v_from_D: float
    v_from_Q: float

    v_to_D: float
    v_to_Q: float
    v_to_a: float
    v_to_b: float
    v_to_c: float
    
    i_D: float
    i_Q: float
    i_a: float
    i_b: float
    i_c: float

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

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, v_from_D:float, v_from_Q:float, v_to_D: float, v_to_Q:float, i_D:float=None, i_Q:float=None):
        if (i_D is None) or (i_Q is None):
            raise ValueError("Have not implemented initial conditions yet.")

        i_a, i_b, i_c = dq02abc(i_D, i_Q, 0, 0)
        v_to_a, v_to_b, v_to_c = dq02abc(v_to_D, v_to_Q, 0, 0)
        
        self.emt_init = InitialConditionsEMT(
            v_from_D=v_from_D,
            v_from_Q=v_from_Q,
            v_to_D=v_to_D,
            v_to_Q=v_to_Q,
            v_to_a=v_to_a,
            v_to_b=v_to_b,
            v_to_c=v_to_c,
            i_D=i_D,
            i_Q=i_Q,
            i_a=i_a,
            i_b=i_b,
            i_c=i_c
        )

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

    def get_derivatives_step_emt_abc(
            self, i_a, i_b, i_c,
            v_from_a, v_from_b, v_from_c, v_to_a, v_to_b, v_to_c):
        wb = self.wbase
        di_bus_a = (wb/self.x_pu) *(v_from_a - v_to_a - self.r_pu * i_a)
        di_bus_b = (wb/self.x_pu) *(v_from_b - v_to_b - self.r_pu * i_b)
        di_bus_c = (wb/self.x_pu) *(v_from_c - v_to_c - self.r_pu * i_c)

        return [di_bus_a, di_bus_b, di_bus_c]
        