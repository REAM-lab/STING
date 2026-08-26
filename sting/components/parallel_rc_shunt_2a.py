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
    # Voltage states
    v_D: float
    v_Q: float
    v_a: float
    v_b: float
    v_c: float
    # Current input
    i_D: float
    i_Q: float


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

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, i_D: float, i_Q:float, v_D:float=None, v_Q:float=None):
        
        if (v_D is None) or (v_Q is None):
            raise ValueError("Have not implemented initial conditions yet.")

        v_a, v_b, v_c = dq02abc(v_D, v_Q, 0, 0)

        self.emt_init = InitialConditionsEMT(
            i_D=i_D, 
            i_Q=i_Q, 
            v_D=v_D, 
            v_Q=v_Q,
            v_a=v_a,
            v_b=v_b,
            v_c=v_c
        )

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

    def get_derivatives_step_emt_abc(self, v_sh_a, v_sh_b, v_sh_c, i_sh_a, i_sh_b, i_sh_c):
        wb = self.wbase

        dv_sh_a = (wb/self.b_pu) * (-v_sh_a*self.g_pu + i_sh_a)
        dv_sh_b = (wb/self.b_pu) * (-v_sh_b*self.g_pu + i_sh_b)
        dv_sh_c = (wb/self.b_pu) * (-v_sh_c*self.g_pu + i_sh_c)

        return [dv_sh_a, dv_sh_b, dv_sh_c]