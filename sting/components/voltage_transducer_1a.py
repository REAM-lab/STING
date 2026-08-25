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

        d/dt v_c2 = (1/tau) * (-v_c2 + v_d^2 + v_q^2)
             v_c = c0*u_one + c1*v_c2 + c2*v_c2**2

        Note: Inputs must be squared voltages. Assumes small deviations
            in voltage and small time constant.
        """
        v2_mag = (v_d**2 + v_q**2)
        ssm = StateSpaceModel(
            A=np.array([[-1/self.tau_s]]),
            B=(1/self.tau_s) * np.array([[1, 1]]),
            C=np.array([[1]]),
            D=np.zeros((1,2)),
            x=DynamicalVariables(name=['v_c'], init=[v2_mag]),
            y=DynamicalVariables(name=['v_c'], init=[v2_mag]),
            u=DynamicalVariables(name=['v_d', 'v_q'], init=[v_d, v_q]),
        )
        return ssm.to_quadratic_bilinear()

    def get_taylor_series_constants(self, v_mag):
        df =  1/(2*v_mag)
        ddf = -1/(8*v_mag**3)

        c0 = v_mag + df*(-v_mag**2) + ddf*(v_mag**4)
        c1 = df - 2*ddf*(v_mag**2)
        c2 = ddf

        return (c0, c1, c2)

    def get_derivatives_step_emt_dq0(self, v_c1, v_d, v_q) -> float:
        u = (v_d**2 + v_q**2)**0.5
        dv_c1 = (1/self.tau_s) * (u - v_c1)

        return [dv_c1]