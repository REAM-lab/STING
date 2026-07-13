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

    kp: float  # Proportional gain
    ki: float  # Integral gain
    kff: float # Feed-forward gain
    xf: float  # Inductive reactance

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, v_out_d, v_out_q, v_d, v_q, i_d, i_q):

        self.emt_init = InitialConditionsEMT(
            z_cc_d = v_out_d - self.kff * v_d + self.xf*i_q,
            z_cc_q = v_out_q - self.kff * v_q - self.xf*i_d
        )

        return self.emt_init

    def get_small_signal_model(self, z_cc_d, z_cc_q):
        
        kp, ki, kff, xf = self.kp, self.ki, self.kff, self.xf

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

    def differential_step_emt_dq0(self, i_ref_d, i_ref_q, i_d, i_q):
        dz_cc_d = self.ki * (i_ref_d - i_d)
        dz_cc_q = self.ki * (i_ref_q - i_q)

        return [dz_cc_d, dz_cc_q]

    def algebraic_step_emt_dq0(self, z_cc_d, z_cc_q, i_ref_d, i_ref_q, i_d, i_q, v_d, v_q):
        
        v_out_d = z_cc_d + self.kp * (i_ref_d - i_d) - self.xf * i_q + self.kff * v_d
        v_out_q = z_cc_q + self.kp * (i_ref_q - i_q) + self.xf * i_d + self.kff * v_q
        
        return [v_out_d, v_out_q]