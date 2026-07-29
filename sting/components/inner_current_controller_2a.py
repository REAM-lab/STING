from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.utils import DynamicalVariables, QuadraticBilinearModel, StateSpaceModel


# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    z_cc_d: float
    z_cc_q: float
    v_d: float
    v_q: float
    i_d: float
    i_q: float
    w: float

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class InnerCurrentController2A:
    """
    Models a second-order inner current controller with the following structure:
                                     ┌──────┐
    v_dq ───────────────────────────▶│ kffv │───────────────┐
                                     └──────┘               │
                  [+]             ┌───────────────┐     [+] ▼[+]
    i_ref_dq ──────▶──┬──────────▶│ PI Controller │──────▶──┼────▶ v_out_dq
                      ▲[-]        └───────────────┘         ▲[+]
                      │                                     │
    i_dq ─────────────┴─────────────▶┌────────┐             │
                                     │ j * xf │─────────────┘
    w ──────────────────────────────▶└────────┘

    where:
    - v_dq: feed-forward voltage in dq frame
    - i_ref_dq: reference current in dq frame
    - i_dq: actual current in dq frame
    - i_out_dq: output current in dq frame

    Parameters:
    - kp_pu: proportional gain in per unit
    - ki_puHz: integral gain in per unit per Hz
    - kffv: feed-forward gain
    - xf_pu: inductive reactance in per unit
    """

    kp_pu: float    # Proportional gain
    ki_puHz: float  # Integral gain
    kffv: float     # Feed-forward gain
    xf_pu: float    # Inductive reactance

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, v_out_d: float, v_out_q: float, v_d: float, v_q: float, i_d: float, i_q: float, w: float):
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.

        - v_out_d [pu]: Output voltage of the inner current controller in d-axis
        - v_out_q [pu]: Output voltage of the inner current controller in q-axis
        - v_d [pu]: Feed-forward voltage in d-axis
        - v_q [pu]: Feed-forward voltage in q-axis
        - i_d [pu]: Target current to be regulated in d-axis
        - i_q [pu]: Target current to be regulated in q-axis
        - w [pu]: frequency
        """

        self.emt_init = InitialConditionsEMT(
            z_cc_d = v_out_d - self.kffv * v_d + self.xf_pu * i_q * w,
            z_cc_q = v_out_q - self.kffv * v_q - self.xf_pu * i_d * w,
            v_d = v_d,
            v_q = v_q,
            i_d = i_d,
            i_q = i_q,
            w = w
        )

        return self.emt_init

    def get_small_signal_model(self, z_cc_d: float, z_cc_q: float, i_d: float, i_q: float, v_d: float, v_q: float, w: float):
        """
        Returns the small-signal state-space model of the inner current controller.

        Inputs:
        - z_cc_d [pu]: Initial value of the state variable associated to integral control block in d-axis
        - z_cc_q [pu]: Initial value of the state variable associated to integral control block in q-axis
        - i_ref_d [pu]: Initial value of the reference current in d-axis
        - i_ref_q [pu]: Initial value of the reference current in q-axis
        - i_d [pu]: Initial value of the actual current or current to be regulated in d-axis
        - i_q [pu]: Initial value of the actual current or current to be regulated in q-axis
        - v_d [pu]: Initial value of the feed-forward voltage in d-axis
        - v_q [pu]: Initial value of the feed-forward voltage in q-axis
        - w [pu]: Initial value of the frequency

        Outputs:
        - StateSpaceModel
        
        Equations to derive the small-signal model:
        dΔz_cc_d/dt = ki * (Δi_ref_d - Δi_d)
        dΔz_cc_q/dt = ki * (Δi_ref_q - Δi_q)
        Δv_out_d = Δz_cc_d + kp * (Δi_ref_d - Δi_d) + kff * Δv_d - xf * (w)ₒ * Δi_q - xf * (i_q)ₒ * Δw
        Δv_out_q = Δz_cc_q + kp * (Δi_ref_q - Δi_q) + kff * Δv_q + xf * (w)ₒ * Δi_d + xf * (i_d)ₒ * Δw

        State vector, input vector, and output vector are:
        x = [Δz_cc_d, Δz_cc_q]
        u = [Δi_ref_d, Δi_ref_q, Δi_d, Δi_q, Δv_d, Δv_q, Δw]
        y = [Δv_out_d, Δv_out_q]

        where:
        - z_cc_d: State variable associated to integral control block in d-axis
        - z_cc_q: State variable associated to integral control block in q-axis
        - i_ref_d: Reference current in d-axis
        - i_ref_q: Reference current in q-axis
        - i_d: Actual current or current to be regulated in d-axis
        - i_q: Actual current or current to be regulated in q-axis
        - v_d: Feed-forward voltage in d-axis
        - v_q: Feed-forward voltage in q-axis
        - w: frequency
        - v_out_d: Output voltage of the inner current controller in d-axis
        - v_out_q: Output voltage of the inner current controller in q-axis     

        State-space representation in tableau form:
        
                │   Δx  │   Δu
        ────────────────────────
        dΔx/dt  │   A   │   B 
        ────────────────────────
        Δy      │   C   │   D

                    │ Δz_cc_d  Δz_cc_q  │   Δi_ref_d   Δi_ref_q  Δi_d     Δi_q         Δv_d    Δv_q  Δw
        ───────────────────────────────────────────────────────────────────────────────────────────────────
        dΔz_cc_d/dt │  0       0        │   ki         0         -ki      0            0       0     0
        dΔz_cc_q/dt │  0       0        │   0          ki        0        -ki          0       0     0  
        ───────────────────────────────────────────────────────────────────────────────────────────────────
        Δv_out_d    │  1       0        │   kp         0         -kp      -xf*(w)ₒ     kff     0     -xf*i_q
        Δv_out_q    │  0       1        │   0          kp        xf*(w)ₒ  -kp          0       kff   xf*i_d
        """
       
        kp, ki, kff, xf = self.kp_pu, self.ki_puHz, self.kffv, self.xf_pu

        A = np.zeros((2,2))
        B = ki * np.hstack([np.eye(2), -np.eye(2), np.zeros((2,3))])
        C = np.eye(2)
        D = np.array([
            [ kp,  0,   -kp,    -xf*w, kff,  0, -xf*i_q],
            [  0, kp,   xf*w,   -kp,  0,    kff, xf*i_d]
        ])

        u = DynamicalVariables(
            name=['i_d_ref', 'i_q_ref', 'i_d', 'i_q', 'v_d', 'v_q', 'w'],
            init=[i_d, i_q, i_d, i_q, v_d, v_q, w]
            )
        x = DynamicalVariables(
            name=['z_cc_d', 'z_cc_q'],
            init=[z_cc_d, z_cc_q]
            )
        
        v_out_d = z_cc_d + self.kffv * v_d - self.xf_pu * i_q * w # Output voltage of the inner current controller in d-axis
        v_out_q = z_cc_q + self.kffv * v_q + self.xf_pu * i_d * w # Output voltage of the inner current controller in q-axis

        y = DynamicalVariables(
            name=['v_out_d', 'v_out_q'],
            init=[v_out_d, v_out_q])

        return StateSpaceModel(A=A, B=B, C=C, D=D, u=u, x=x, y=y)

    def get_quadratic_bilinear_model(self, z_cc_d: float, z_cc_q: float, i_d: float, i_q: float, v_d: float, v_q: float, w: float):
        """
        Returns the quadratic bilinear model of the inner current controller.
        NOTE: Unlike the small-signal model, w*i_dq is a model input

        QBM Inputs:
        - z_cc_d [pu]: State variable associated to integral control block in d-axis
        - z_cc_q [pu]: State variable associated to integral control block in q-axis
        - i_ref_d [pu]: Reference current in d-axis
        - i_ref_q [pu]: Reference current in q-axis
        - i_d [pu]: Actual current or current to be regulated in d-axis
        - i_q [pu]: Actual current or current to be regulated in q-axis
        - v_d [pu]: Feed-forward voltage in d-axis
        - v_q [pu]: Feed-forward voltage in q-axis
        - w*i_d [pu]: frequency *TIMES* current to be regulated in d-axis
        - w*i_q [pu]: frequency *TIMES* current to be regulated in q-axis

        QBM Outputs:
        - v_out_d [pu]: Output voltage of the inner current controller in d-axis
        - v_out_q [pu]: Output voltage of the inner current controller in q-axis
        """
        
        kp, ki, kff, xf = self.kp_pu, self.ki_puHz, self.kffv, self.xf_pu

        A = np.zeros((2,2))
        B = ki * np.hstack([np.eye(2), -np.eye(2), np.zeros((2,4))])
        C = np.eye(2)
        D = np.array([
            [ kp,  0,-kp,  0, kff,  0,  0,-xf],
            [  0, kp,  0,-kp,  0, kff, xf,  0]
        ])

        H = np.zeros((2, 4))
        N = np.zeros((2, 16))

        u = DynamicalVariables(
            name=['i_d_ref', 'i_q_ref', 'i_d', 'i_q', 'v_d', 'v_q', 'w*i_d', 'w*i_q'],
            init=[i_d, i_q, i_d, i_q, v_d, v_q, w*i_d, w*i_q]
            )
        y = DynamicalVariables(name=['v_out_d', 'v_out_q'])
        x = DynamicalVariables(
            name=['z_cc_d', 'z_cc_q'],
            init= [z_cc_d, z_cc_q]
        ) 

        return QuadraticBilinearModel(A=A, B=B, C=C, D=D, N=N, H=H, x=x, y=y, u=u)

        

    def define_variables_emt_dq0(self):

        x = DynamicalVariables(
            name = ['z_cc_d', 'z_cc_q'],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.z_cc_d, self.emt_init.z_cc_q]
        )

        u = DynamicalVariables(
            name=["i_d_ref", "i_q_ref", "i_d", "i_q", "v_d", "v_q"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.i_d, self.emt_init.i_q, self.emt_init.i_d, self.emt_init.i_q, self.emt_init.v_d, self.emt_init.v_q]

        )

        return [x, u]

    def get_derivatives_step_emt_dq0(self, i_ref_d: float, i_ref_q: float, i_d: float, i_q: float):
        """
        Returns the derivatives of the state variables for the EMT simulation step.

        Inputs:
        - i_ref_d [pu]: Reference current in d-axis
        - i_ref_q [pu]: Reference current in q-axis
        - i_d [pu]: Actual current or current to be regulated in d-axis
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
        Returns the current outputs of the inner current controller for the EMT simulation step. These current outputs can be used as current references.
        
        EMT Inputs:
        - z_cc_d [pu]: State variable associated to integral control block in d-axis
        - z_cc_q [pu]: State variable associated to integral control block in q-axis
        - i_ref_d [pu]: Reference current in d-axis
        - i_ref_q [pu]: Reference current in q-axis
        - i_d [pu]: Actual current or current to be regulated in d-axis
        - i_q [pu]: Actual current or current to be regulated in q-axis
        - v_d [pu]: Feed-forward voltage in d-axis
        - v_q [pu]: Feed-forward voltage in q-axis
        - w [pu]: frequency

        EMT Outputs:
        - v_out_d [pu]: Output voltage of the inner current controller in d-axis
        - v_out_q [pu]: Output voltage of the inner current controller in q-axis
        """ 

        # Compute output of PI controller in d-axis and q-axis
        out_pi_d = z_cc_d + self.kp_pu * (i_ref_d - i_d)
        out_pi_q = z_cc_q + self.kp_pu * (i_ref_q - i_q)

        # Compute output voltage in d-axis and q-axis
        v_out_d = out_pi_d + self.kffv * v_d - self.xf_pu * i_q * w
        v_out_q = out_pi_q + self.kffv * v_q + self.xf_pu * i_d * w
        
        return [v_out_d, v_out_q]