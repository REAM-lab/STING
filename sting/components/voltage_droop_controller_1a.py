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

    q_f: float

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class VoltageDroopController1A:
    """
    Models a voltage droop control of the structure shown below:
    """

    k_q_pu: float  # Droop gain in per unit
    w_q_puHz: float  # Time constant of the low-pass filter in per unit per Hz

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, q: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Inputs:
        - q [pu]: Steady-state reactive power
         
        Outputs:
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            q_f = q
        )

        return self.emt_init

    def get_derivatives_step_emt_dq0(self, q: float, q_f: float) -> float:
        """
        Returns the derivatives of the states of the voltage droop control for the EMT simulation step.
        
        Inputs:
        - q [pu]: Reactive power
        - q_f [pu]: State variable associated to low-pass filter
        
        Outputs:
        - d_q_f [pu/s]: Derivative of the state variable associated to low-pass filter
        """

        # Compute derivative of the state variable associated to low-pass filter
        d_q_f = self.w_q_puHz * (q - q_f)

        return d_q_f

    def get_algebraics_step_emt_dq0(self, v_ref: float, q_ref: float, q_f: float) -> list[float]:
        """
        Returns the algebraic outputs of the voltage droop control for the EMT simulation step.
        """

        # Compute reference voltage in d_axis
        v_d_ref = v_ref + self.k_q_pu * (q_ref - q_f)

        # Fix the reference voltage in q_axis to zero
        v_q_ref = 0.0

        return [v_d_ref, v_q_ref]