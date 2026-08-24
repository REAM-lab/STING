from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel


class InitialConditionsEMT(NamedTuple):
    x_gov: float


@dataclass(slots=True)
class Governor1A:
    """
    A 1st order model of a governor with speed-droop.
    
           ┌──────┐  [-]       ┌───────────────┐
    Δω ───▶│ 1/kr │───▶──┬────▶│ 1/(1 + tau*s) │───▶ Δx
           └──────┘      ▲[+]  └───────────────┘
                         │
                        p_ref

    The dynamics of the governor are given by:
        tau * d/dt Δx = p_ref - (Δω / kr) - Δx

    where the inputs and states are defined as follows:
        - p_ref: Load reference or active power setpoint.
        - Δω: Angular velocity deviation from nominal.
        - Δx: Control valve position/height/power
    """
    tau_s: float # Time constant (in seconds)
    kr_pu: float # Speed regulator gain (in pu)

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, p_ref:float, w:float) -> InitialConditionsEMT:
        self.emt_init = InitialConditionsEMT(x_gov=p_ref - (1/self.kr_pu) * w)
        return self.emt_init

    def get_small_signal_model(self, x_gov:float, p_ref:float, w:float):
        ssm = StateSpaceModel(
            A=np.array([[-1/self.tau_s]]),
            B=(1/self.tau_s) *np.array([[1, -1/self.kr_pu]]),
            C=np.array([[1]]),
            D=np.array([[0]]),
            x=DynamicalVariables(name=['x_gov'], init=[x_gov]),
            u=DynamicalVariables(name=['p_ref', 'w'], init=[p_ref, w]),
            y=DynamicalVariables(name=['x_gov'], init=[x_gov]),
        )
        return ssm

    def get_quadratic_bilinear_model(self, x_gov:float, p_ref:float, w:float):
        ssm = self.get_small_signal_model(x_gov, p_ref, w)
        return ssm.to_quadratic_bilinear()

    def get_derivatives_step_emt(self, x_gov:float, p_ref:float, w:float) -> float:
        dx_gov = (1/self.tau_s) * (p_ref - (1/self.kr_pu) * w - x_gov)
        return np.array([dx_gov])
