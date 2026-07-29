# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple

# -------------------------------------------------------------------------
# Import sting code
# -------------------------------------------------------------------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    """Store the initial conditions of the reactive power controller for the EMT simulation."""

    q_ref: float
    i_ref_q: float
    z_pi: float


# ---------------------------------------
# Main class
# ---------------------------------------
@dataclass(slots=True)
class ReactivePowerPI1A:
    kp_pu: float
    ki_puHz: float
    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, q_ref: float, i_ref_q: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Inputs:
        - q_ref [pu]: Steady-state active power
        - i_ref_q [pu]: Steady-state d-axis current reference
         
        Outputs:
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            q_ref = q_ref,
            i_ref_q = i_ref_q,
            z_pi = i_ref_q
        )

        return self.emt_init

    def get_small_signal_model(self, z_pi, q_ref):
        ssm = StateSpaceModel(
            A = np.array([[0]]),
            B = self.ki_puHz * np.array([[1, -1]]),
            C = np.array([[1]]),
            D = self.kp_pu * np.array([[1, -1]]),
            x = DynamicalVariables(name=['z_pi'], init=z_pi),
            u = DynamicalVariables(name=["q_ref", "q"], init = [q_ref, q_ref]),
            y = DynamicalVariables(name=['i_q_ref'])
        )
        return ssm

    def define_variables_emt_abc(self):

        # States 
        x = DynamicalVariables(
            name = ['z_pi'],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.z_pi]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["q_ref", "q"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.q_ref, self.emt_init.q_ref]
        )

        return [x, u]

    def get_derivatives_step_emt_abc(self, q_ref: float, q: float, z_pi: float) -> float:
        """
        Returns the derivatives of the states of the reactive power controller for the EMT simulation step.
        
        Inputs:
        - q_ref [pu]: Reactive power reference
        - q [pu]: Reactive power
        - z_pi [pu]: State variable associated to the PI controller
        
        Outputs:
        - d_z_pi [pu/s]: Derivative of the state variable associated to the PI controller
        """

        # Compute derivative of the state variable associated to the PI controller
        d_z_pi = self.ki_puHz * (-1) * (q_ref - q)

        return d_z_pi

    def get_algebraics_step_emt_abc(self, q_ref: float, q: float, z_pi: float) -> list[float]:
        """
        Returns the algebraic outputs of the reactive power controller for the EMT simulation step.

        Inputs:
        - q_ref [pu]: Reactive power reference
        - q [pu]: Reactive power
        - z_pi [pu]: State variable associated to the PI controller

        Outputs:
        - i_ref_q [pu]: q-axis current reference
        """

        # Compute q-axis current reference
        i_ref_q = self.kp_pu * (-1) * (q_ref - q) + z_pi

        return i_ref_q
