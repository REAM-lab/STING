import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

class InitialConditionsEMT(NamedTuple):
    z_cc_d: float
    z_cc_q: float


@dataclass(slots=True)
class InnerCurrentController:
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

    def get_steady_state(self, v_vsc_d, v_vsc_q, v_d, v_q, i_d, i_q):

        self.emt_init = InitialConditionsEMT(
            z_d = v_vsc_d - self.kff * v_d + self.xf*i_d,
            z_q = v_vsc_q - self.kff * v_q - self.xf*i_q
        )

        return self.emt_init

    def get_small_signal_model(self, z_d, z_q):
        
        kp, ki, kff, xf = self.kp, self.ki, self.kff, self.xf

        A = np.zeros((2,2))
        B = ki * np.hstack([np.eye(2), -np.eye(2), np.zeros((2,2))])
        C = np.eye(2)
        D = np.array([
            [ kp,  0,-kp,-xf, kff,  0],
            [  0, kp, xf,-kp,  0, kff]
        ])

        """
        u = DynamicalVariables(name=['i_bus_d_ref', 'i_bus_q_ref', 'i_bus_d', 'i_bus_q']), 
                                          y = DynamicalVariables(name=['e_d', 'e_q']),
                                          x = DynamicalVariables(   name=['pi_cc_d', 'pi_cc_q'],
                                                                    init= [pi_cc_d, pi_cc_q]) )
        """

    def differential_step_emt_dq0(self, ref_d, ref_q, i_d, i_q):
        dz_cc_d = self.ki * (ref_d - i_d)
        dz_cc_q = self.ki * (ref_q - i_q)

        return [dz_cc_d, dz_cc_q]

    def algebraic_step_emt_dq0(self, z_cc_d, z_cc_q, ref_d, ref_q, i_d, i_q, v_d, v_q):
        
        v_vsc_d = z_cc_d + self.kp * (ref_d - i_d) - self.xf * i_q + self.kff * v_d
        v_vsc_q = z_cc_q + self.kp * (ref_q - i_q) + self.xf * i_d + self.kff * v_q
        
        return [v_vsc_d, v_vsc_q]