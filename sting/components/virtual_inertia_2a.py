# ------------------------------------------------------------------------------
# Import libraries
# ------------------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

# ---------------------------------------
# Subclasses
# ---------------------------------------
class InitialConditionsEMT(NamedTuple):
    angle: float

# ---------------------------------------
# Main class
# ---------------------------------------
@dataclass(slots=True)
class VirtualInertia2A:
    """
    The virtual inertia model is a second-order model that emulates the dynamics of a synchronous generator.

    Parameters:
    - kd_w_pu: damping gain [pu] of the active power controller
    - h_sec: virtual inertia [s]
    - w_nom: nominal frequency [rad/s] of the system

    """
    kd_w_pu: float
    h_sec: float
    w_nom: float

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, angle: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Inputs:
        - angle [rad]: Steady-state angle of the active power controller
         
        Outputs:
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            angle = angle,
            w = 1.0
        )

        return self.emt_init

    def get_differential_step_emt(self, w: float, p_ref: float, p: float) -> list[float]:
        """
        Compute the derivates with respect to time of the states of the virtual inertia model
        for the next time step in the EMT simulation.

        Inputs:
        - w [pu]: angular frequency. It is a state of the virtual inertia model.
        - p_ref [pu]: reference active power. It is an input to the virtual inertia model.
        - p [pu]: measured active power. It is an input to the virtual inertia model.

        Outputs:
        - d_angle [rad/s]: derivative of the angle of the active power controller
        - d_w [pu/s]: derivative of the angular frequency of the active power controller
        """
    
        # Extract the list of parameters
        kd_w = self.kd_w_pu  # damping gain of active power controller
        h = self.h_sec  # virtual inertia
        w_nom = self.w_nom  # nominal frequency of the system

        # Derivative of the angle
        d_angle_pc = w_nom * w
        
        # Derivative of the angular frequency
        d_w_pc = 1/(2 * h) * (p_ref - p - kd_w * (w - 1))
    
        return [d_angle_pc, d_w_pc]