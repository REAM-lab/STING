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
    """Store the initial conditions of the active power controller for the EMT simulation."""

    p_ref: float
    i_ref_d: float


# ---------------------------------------
# Main class
# ---------------------------------------
@dataclass(slots=True)
class ActivePowerPI1A:
    kp_pu: float
    ki_puHz: float
    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, p_ref: float, i_ref_d: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Inputs:
        - p_ref [pu]: Steady-state active power
        - i_ref_d [pu]: Steady-state d-axis current reference
         
        Outputs:
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            p_ref = p_ref,
            i_ref_d = i_ref_d
        )

        return self.emt_init

    def define_variables_emt_abc(self):

        # States 
        x = DynamicalVariables(
            name = ['z_pi'],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.z_pi]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["p_ref", "p"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.p_ref, self.emt_init.p_ref]
        )

        return [x, u]

    def get_derivatives_step_emt_abc(self, p_ref: float, p: float, z_pi: float) -> float:
        """
        Returns the derivatives of the states of the active power controller for the EMT simulation step.
        
        Inputs:
        - p_ref [pu]: Active power reference
        - p [pu]: Active power
        - z_pi [pu]: State variable associated to the PI controller
        
        Outputs:
        - d_z_pi [pu/s]: Derivative of the state variable associated to the PI controller
        """

        # Compute derivative of the state variable associated to the PI controller
        d_z_pi = self.ki_puHz * (p_ref - p)

        return d_z_pi

    def get_algebraics_step_emt_abc(self, p_ref: float, p: float, z_pi: float) -> list[float]:
        """
        Returns the algebraic outputs of the active power controller for the EMT simulation step.

        Inputs:
        - p_ref [pu]: Active power reference
        - p [pu]: Active power
        - z_pi [pu]: State variable associated to the PI controller

        Outputs:
        - i_ref_d [pu]: d-axis current reference
        """

        # Compute d-axis current reference
        i_ref_d = self.kp_pu * (p_ref - p) + z_pi

        return [i_ref_d]
