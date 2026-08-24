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
    A 1st order model of a voltage transducer with a low-pass filter and 
    no load compensation.

             ┌─────────────────────┐    ┌─────────────┐
    v_dq ───▶│ (v_d^2 + v_q^2)^0.5 │───▶│ 1 / 1+s*tau │───▶ v_c
             └─────────────────────┘    └─────────────┘
    """
    tau_s: float # Filter time constant, in seconds.

    emt_init: InitialConditionsEMT = field(init=False)


    def get_steady_state(self, v_d, v_q):
        v_c1 = ( v_d**2 + v_q**2 )**0.5

        self.emt_init = InitialConditionsEMT(v_c1=v_c1)

    def get_small_signal_model(self, v_d, v_q):
        """
        d/dt v_c1 = (1/tau) * [-v_c1 + (v_d0 * Δv_d + v_q0 * Δv_q ) / v_mag0]
        """
        v_mag = (v_d**2 + v_q**2)**0.5
        ssm = StateSpaceModel(
            A=np.array([[-1/self.tau_s]]),
            B=(self.tau_s * v_mag)**-1 * np.array([[v_d, v_q]]),
            C=np.array([[1]]),
            D=np.zeros((1,2)),
            x=DynamicalVariables(name=['v_c'], init=[v_mag]),
            y=DynamicalVariables(name=['v_c'], init=[v_mag]),
            u=DynamicalVariables(name=['v_d', 'v_q'], init=[v_d, v_q]),
        )

        return ssm

    def get_quadratic_bilinear_model(self, v_d, v_q):
        """
        The contents of this function should not be presented as original work by another author.

                δ = v_d^2 + v_q^2 - v_mag^2
        d/dt v_c1 = (1/tau) * (-v_c1 + v_mag + δ / (2*v_mag) )
        
        Note: Inputs must be squared voltages.
        """
        v_mag = (v_d**2 + v_q**2)**0.5
        c1 = 1/(2*v_mag)
        c0 = v_mag - c1*v_mag**2

        ssm = StateSpaceModel(
            A=np.array([[-1/self.tau_s]]),
            B=(1/self.tau_s) * np.array([[c0, c1, c1]]),
            C=np.array([[1]]),
            D=np.zeros((1,3)),
            x=DynamicalVariables(name=['v_c'], init=[v_mag]),
            y=DynamicalVariables(name=['v_c'], init=[v_mag]),
            u=DynamicalVariables(name=['one', 'v_d^2', 'v_q^2'], init=[1, v_d, v_q]),
        )
        return ssm.to_quadratic_bilinear()

    def get_derivatives_step_emt_dq0(self, v_c1, v_d, v_q) -> float:
        u = (v_d**2 + v_q**2)**0.5
        dv_c1 = (1/self.tau_s) * (u - v_c1)

        return np.array([dv_c1])