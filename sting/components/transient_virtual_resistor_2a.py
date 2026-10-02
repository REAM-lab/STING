import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple

from sting.utils.dynamical_systems import (
    StateSpaceModel,
    DynamicalVariables)


class InitialConditionsEMT(NamedTuple):
    """Store the initial conditions of the TVR for the EMT simulation."""
    z_tvr_d: float
    z_tvr_q: float


@dataclass(slots=True)
class TransientVirtualResistor2A:
    w_TVR_pu: float
    R_v_pu: float

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, i_d: float, i_q: float) -> InitialConditionsEMT:

        self.emt_init = InitialConditionsEMT(
            z_tvr_d=self.R_v_pu * i_d,
            z_tvr_q=self.R_v_pu * i_q,
        )

        return self.emt_init

    def get_derivatives_step_emt_dq0(self, i_d: float, i_q: float, z_tvr_d: float, z_tvr_q: float) -> list[float]:
        
        w_TVR = self.w_TVR_pu
        R_v = self.R_v_pu

        d_z_tvr_d = w_TVR * (R_v * i_d - z_tvr_d)
        d_z_tvr_q = w_TVR * (R_v * i_q - z_tvr_q)

        return [d_z_tvr_d, d_z_tvr_q]

    def get_algebraics_step_emt_dq0(self, v_ref: float, i_d: float, i_q: float, z_tvr_d: float, z_tvr_q: float) -> list[float]:

        R_v = self.R_v_pu

        delta_v_d = R_v * i_d - z_tvr_d
        delta_v_q = R_v * i_q - z_tvr_q

        v_d = v_ref - delta_v_d
        v_q = - delta_v_q

        return [v_d, v_q]

    
    def get_small_signal_model(self, i_d: float, i_q: float, v_ref: float):

        w_TVR = self.w_TVR_pu
        R_v = self.R_v_pu

        A = np.array([
            [-w_TVR,  0     ],
            [0,      -w_TVR ],
        ])

        B = np.array([
            [0, w_TVR * R_v, 0],
            [0, 0, w_TVR * R_v],
        ])

        C = np.array([
            [1,  0],
            [0,  1],
        ])

        D = np.array([
            [1, -R_v, 0],
            [0, 0, -R_v],
        ])

        ssm = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            x=DynamicalVariables(
                name=["z_tvr_d", "z_tvr_q"],
                init=[R_v * i_d, R_v * i_q],
            ),
            u=DynamicalVariables(
                name=["v_ref", "i_d", "i_q"],
                init=[v_ref, i_d, i_q],
            ),
            y=DynamicalVariables(
                name=["v_d", "v_q"],
                init=[v_ref, 0],
            ),
        )

        return ssm