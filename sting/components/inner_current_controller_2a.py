import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

class InitialConditionsEMT(NamedTuple):
    z_cc_d: float
    z_cc_q: float


@dataclass(slots=True)
class InnerCurrentController2A:
    """

    Inputs 
    i_ref: Reference current
    i_dq: Current in dq
    v_dq: Feed-forward voltage in dq


    TODO: Add w as an input
    """

    kp_pu: float  # Proportional gain
    ki_puHz: float  # Integral gain
    kffv: float # Feed-forward gain
    xf_pu: float  # Inductive reactance

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, v_out_d: float, v_out_q: float, v_d: float, v_q: float, i_d: float, i_q: float, w: float):
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Inputs:
        - v_out_d [pu]: Output voltage in d-axis
        - v_out_q [pu]: Output voltage in q-axis
        - v_d [pu]: Feed-forward voltage in d-axis
        - v_q [pu]: Feed-forward voltage in q-axis
        - i_d [pu]: Actual current in d-axis
        - i_q [pu]: Actual current in q-axis
        - w [pu]: frequency
         
        Outputs:
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            z_cc_d = v_out_d - self.kffv * v_d + self.xf_pu * i_q * w,
            z_cc_q = v_out_q - self.kffv * v_q - self.xf_pu * i_d * w
        )

        return self.emt_init

    def get_small_signal_model(self, z_cc_d: float, z_cc_q: float):
        """
        Returns the small-signal state-space model of the inner current controller.
        TODO: Add w as an input
        Inputs:
        - z_cc_d [pu]: State associated to integral control block in d-axis
        - z_cc_q [pu]: State associated to integral control block in q-axis
        
        Outputs:
        - ssm: Small-signal state-space model of the inner current controller
        """
        
        kp, ki, kff, xf = self.kp_pu, self.ki_puHz, self.kffv, self.xf_pu

        A = np.zeros((2,2))
        B = ki * np.hstack([np.eye(2), -np.eye(2), np.zeros((2,2))])
        C = np.eye(2)
        D = np.array([
            [ kp,  0,-kp,-xf, kff,  0],
            [  0, kp, xf,-kp,  0, kff]
        ])

        ssm = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            u = DynamicalVariables(name=['i_cc_d_ref', 'i_cc_q_ref', 'i_cc_d', 'i_cc_q', 'v_cc_d', 'v_cc_q']), 
            y = DynamicalVariables(name=['v_out_d', 'v_out_q']),
            x = DynamicalVariables(
                name=['z_cc_d', 'z_cc_q'],
                init= [z_cc_d, z_cc_q]
            ) 
        )
        return ssm

    def get_derivatives_step_emt_dq0(self, i_ref_d: float, i_ref_q: float, i_d: float, i_q: float):
        """
        Returns the derivatives of the state variables for the EMT simulation step.

        Inputs:
        - i_ref_d [pu]: Reference current in d-axis
        - i_ref_q [pu]: Reference current in q-axis
        - i_d [pu]: Actual current in d-axis
        - i_q [pu]: Actual current in q-axis

        Outputs:
        - d_z_cc_d: Derivative of the state associated to integral control block in d-axis
        - d_z_cc_q: Derivative of the state associated to integral control block in q-axis
        """

        d_z_cc_d = self.ki_puHz * (i_ref_d - i_d)
        d_z_cc_q = self.ki_puHz * (i_ref_q - i_q)

        return [d_z_cc_d, d_z_cc_q]

    def get_algebraics_step_emt_dq0(self, z_cc_d: float, z_cc_q: float, i_ref_d: float, i_ref_q: float, i_d: float, i_q: float, v_d: float, v_q: float, w: float):
        """
        Returns the voltage outputs of the inner current controller for the EMT simulation step. These voltage outputs can be used as voltage references.
        Inputs:
        - z_cc_d [pu]: State variable z_cc_d
        - z_cc_q [pu]: State variable z_cc_q
        - i_ref_d [pu]: Reference current in d-axis
        - i_ref_q [pu]: Reference current in q-axis
        - i_d [pu]: Actual current in d-axis
        - i_q [pu]: Actual current in q-axis
        - v_d [pu]: Feed-forward voltage in d-axis
        - v_q [pu]: Feed-forward voltage in q-axis
        - w [pu]: frequency

        Outputs:
        - v_out_d [pu]: Output voltage in d-axis
        - v_out_q [pu]: Output voltage in q-axis
        """ 

        # Compute output of PI controller in d-axis and q-axis
        out_pi_d = z_cc_d + self.kp_pu * (i_ref_d - i_d)
        out_pi_q = z_cc_q + self.kp_pu * (i_ref_q - i_q)

        v_out_d = out_pi_d + self.kffv * v_d - self.xf_pu * i_q * w
        v_out_q = out_pi_q + self.kffv * v_q + self.xf_pu * i_d * w
        
        return [v_out_d, v_out_q]