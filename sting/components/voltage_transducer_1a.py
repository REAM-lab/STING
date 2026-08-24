from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.utils.dynamical_systems import DynamicalVariables, QuadraticBilinearModel, StateSpaceModel


# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    v_c1: float

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class VoltageTransducer1A:
    """
    First order model of a voltage transducer with a low-pass filter.

    v_dq ───▶┌──────────────────────────────────┐    ┌─────────────┐
    i_dq ───▶│ ‖ v_dq + (r_c + jx_c) * i_dq ‖_2 │───▶│ 1 / 1+s*t_r │───▶ v_c
             └──────────────────────────────────┘    └─────────────┘

    TODO: Add current inputs to SSM
    """
    t_r: float
    r_c: float
    x_c: float       

    emt_init: InitialConditionsEMT = field(init=False)


    def get_steady_state(self, v_d, v_q, i_d, i_q):
        v_c1 = ( (v_d + self.r_c*i_d - self.x_c*i_q)**2 + (v_q + self.r_c*i_q + self.x_c*i_d)**2 )**0.5

        self.emt_init = InitialConditionsEMT(v_c1=v_c1)

    def get_small_signal_model(self, v_d, v_q, i_d, i_q):
        """
        NOTE: Model does not currently perform load compensation!

        d/dt v_c1 = (1/t_r) * [-v_c1 + (v_d0 * Δv_d + v_q0 * Δv_q ) / v_mag0]
        """
        v_mag = (v_d**2 + v_q**2)**0.5
        ssm = StateSpaceModel(
            A=np.array([[-1/self.t_r]]),
            B=(self.t_r * v_mag)**-1 * np.array([[v_d, v_q, 0, 0]]),
            C=np.array([[1]]),
            D=np.zeros((1,4)),
            x=DynamicalVariables(name=['v_c'], init=[v_mag]),
            y=DynamicalVariables(name=['v_c'], init=[v_mag]),
            u=DynamicalVariables(name=['v_d', 'v_q', 'i_d', 'i_q'], init=[v_d, v_q, i_d, i_q]),
        )

        return ssm

    def get_quadratic_bilinear_model(self, ):
        pass

    def get_derivatives_step_emt_dq0(self, v_c1, v_d, v_q, i_d, i_q) -> float:

        u = ( (v_d + self.r_c*i_d - self.x_c*i_q)**2 + (v_q + self.r_c*i_q + self.x_c*i_d)**2 )**0.5

        dv_c1 = (1/self.t_r) * (u - v_c1)

        return np.array([dv_c1])