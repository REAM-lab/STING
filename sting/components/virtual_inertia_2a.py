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
    angle_rad: float
    w_pu: float
    p_ref: float
    p_sh: float

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


    def get_differential_step_emt(self, w_pc: float, p_ref: float, p_sh: float) -> list:
        """
        Compute the derivates with respect to time of the states of the virtual inertia model
        for the next time step in the EMT simulation.
        Inputs:
        - w_pc: angular frequency of the active power controller [pu]. It is a state of the virtual inertia model.
        - p_ref: reference active power [pu]. It is an input to the virtual inertia model.
        - p_sh: measured active power at the shunt of the LCL filter [pu]. It is an input to the virtual inertia model.
        Outputs:
        - d_angle_pc: derivative of the angle of the active power controller [rad/s]
        - d_w_pc: derivative of the angular frequency of the active power controller [pu/s]
        """
    
        # Extract the list of parameters
        kd_w = self.kd_w_pu  # damping gain of active power controller
        h = self.h_sec  # virtual inertia
        w_nom = self.w_nom  # nominal frequency of the system

        # Derivative of the angle
        d_angle_pc = w_nom * w_pc
        
        # Derivative of the angular frequency
        d_w_pc = 1/(2 * h) * (p_ref - p_sh - kd_w * (w_pc - 1))
    
        return [d_angle_pc, d_w_pc]