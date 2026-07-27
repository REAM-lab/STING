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
    z_vc_d: float
    z_vc_q: float

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class InnerVoltageController2A:
    """
    Models an inner voltage control of the structure shown below: 

    i_dq --------> [kffi] ---------------------> [+1] ------
                                                            |
    v_ref_dq ----> [+1] --- + -----> [PI controller] ------ + ---> i_out_dq
                            |                               |
                           [-1]                             |
                            |                               |
    v_dq --------------------- x --->[j cf] ----> [+1] -----
    w -------------------------|

    where:
    - i_dq: feed-forward voltage in dq frame
    - v_ref_dq: reference voltage in dq frame
    - v_dq: actual voltage in dq frame
    - i_out_dq: output current in dq frame

    Parameters:
    - kp_pu: proportional gain in per unit
    - ki_puHz: integral gain in per unit per Hz
    - kffv: feed-forward gain
    - cf_pu: capacitance in per unit


    TODO: Add w as an input in small signal model
    """

    kp_pu: float  # Proportional gain
    ki_puHz: float  # Integral gain
    kffi: float # Feed-forward gain
    cf_pu: float  # Inductive reactance

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, i_out_d: float, i_out_q: float, i_d: float, i_q: float, v_d: float, v_q: float, w: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Inputs:
        - i_out_d [pu]: Output current of the inner voltage controller in d-axis
        - i_out_q [pu]: Output current of the inner voltage controller in q-axis
        - i_d [pu]: Feed-forward current in d-axis
        - i_q [pu]: Feed-forward current in q-axis
        - v_d [pu]: Target voltage to be regulated in d-axis
        - v_q [pu]: Target voltage to be regulated in q-axis
        - w [pu]: frequency
         
        Outputs:
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            z_vc_d = i_out_d - self.kffi * i_d + self.cf_pu * v_q * w,
            z_vc_q = i_out_q - self.kffi * i_q - self.cf_pu * v_d * w
        )

        return self.emt_init

    def get_derivatives_step_emt_dq0(self, v_ref_d: float, v_ref_q: float, v_d: float, v_q: float) -> list[float]:
        """
        Returns the derivatives of the state variables for the EMT simulation step.

        Inputs:
        - v_ref_d [pu]: Reference voltage in d-axis
        - v_ref_q [pu]: Reference voltage in q-axis
        - v_d [pu]: Target voltage to be regulated in d-axis
        - v_q [pu]: Target voltage to be regulated in q-axis

        Outputs:
        - d_z_vc_d: Derivative of the state associated to integral control block in d-axis
        - d_z_vc_q: Derivative of the state associated to integral control block in q-axis
        """

        d_z_vc_d = self.ki_puHz * (v_ref_d - v_d)
        d_z_vc_q = self.ki_puHz * (v_ref_q - v_q)

        return [d_z_vc_d, d_z_vc_q]

    def get_algebraics_step_emt_dq0(self, z_vc_d: float, z_vc_q: float, v_ref_d: float, v_ref_q: float, v_d: float, v_q: float, i_d: float, i_q: float, w: float) -> list[float]:
        """
        Returns the current outputs of the inner current controller for the EMT simulation step. These voltage outputs can be used as voltage references.
        Inputs:
        - z_vc_d [pu]: State variable associated to integral control block in d-axis
        - z_vc_q [pu]: State variable associated to integral control block in q-axis
        - v_ref_d [pu]: Reference voltage in d-axis
        - v_ref_q [pu]: Reference voltage in q-axis
        - v_d [pu]: Target voltage to be regulated in d-axis
        - v_q [pu]: Target voltage to be regulated in q-axis
        - i_d [pu]: Feed-forward current in d-axis
        - i_q [pu]: Feed-forward current in q-axis
        - w [pu]: frequency

        Outputs:
        - i_out_d [pu]: Output current of the inner voltage controller in d-axis
        - i_out_q [pu]: Output current of the inner voltage controller in q-axis
        """ 

        # Compute output of PI controller in d-axis and q-axis
        out_pi_d = z_vc_d + self.kp_pu * (v_ref_d - v_d)
        out_pi_q = z_vc_q + self.kp_pu * (v_ref_q - v_q)

        # Compute output current in d-axis and q-axis
        i_out_d = out_pi_d + self.kffi * i_d - self.cf_pu * v_q * w
        i_out_q = out_pi_q + self.kffi * i_q + self.cf_pu * v_d * w
        
        return [i_out_d, i_out_q]