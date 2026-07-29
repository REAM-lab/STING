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
    Models a second-order inner voltage controller with the following structure: 
                                     ┌──────┐
    i_dq ───────────────────────────▶│ kffi │───────────────┐
                                     └──────┘               │
                  [+]             ┌───────────────┐     [+] ▼[+]
    v_ref_dq ──────▶──┬──────────▶│ PI Controller │──────▶──┼────▶ i_out_dq
                      ▲[-]        └───────────────┘         ▲[+]
                      │                                     │
    v_dq ─────────────┴─────────────▶┌────────┐             │
                                     │ j * cf │─────────────┘
    w ──────────────────────────────▶└────────┘

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

    def get_small_signal_model(self, z_vc_d: float, z_vc_q: float, v_d: float, v_q: float, i_d: float, i_q: float, w: float):
        """
        Returns the small-signal model of the inner voltage controller.

        Inputs:
        - z_vc_d [pu]: Initial value of the state variable associated to integral control block in d-axis
        - z_vc_q [pu]: Initial value of the state variable associated to integral control block in q-axis
        - v_d [pu]: Initial value of actual voltage to be regulated in d-axis
        - v_q [pu]: Initial value of actual voltage to be regulated in q-axis
        - i_d [pu]: Initial value of feed-forward current in d-axis
        - i_q [pu]: Initial value of feed-forward current in q-axis
        - w [pu]: Initial value of frequency

        Outputs:
        - ssm: State-space model object.

        Equations to derive the small-signal model:
        dΔz_vc_d/dt = ki * (Δv_ref_d - Δv_d)
        dΔz_vc_q/dt = ki * (Δv_ref_q - Δv_q)
        Δi_out_d = Δz_vc_d + kp * (Δv_ref_d - Δv_d) + kff * Δi_d - cf * (w)ₒ * Δv_q - cf * (v_q)ₒ * Δw
        Δi_out_q = Δz_vc_q + kp * (Δv_ref_q - Δv_q) + kff * Δi_q + cf * (w)ₒ * Δv_d + cf * (v_d)ₒ * Δw
        
        where:
        - z_vc_d: State variable associated to integral control block in d-axis
        - z_vc_q: State variable associated to integral control block in q-axis
        - v_ref_d: Reference voltage in d-axis
        - v_ref_q: Reference voltage in q-axis
        - v_d: Actual voltage to be regulated in d-axis
        - v_q: Actual voltage to be regulated in q-axis
        - i_d: Feed-forward current in d-axis
        - i_q: Feed-forward current in q-axis
        - w: frequency
        - i_out_d: Output current of the inner voltage controller in d-axis
        - i_out_q: Output current of the inner voltage controller in q-axis

        State vector, input vector, and output vector are:
        Δx = [Δz_vc_d, Δz_vc_q]
        Δu = [Δv_ref_d, Δv_ref_q, Δv_d, Δv_q, Δi_d, Δi_q, Δw]
        Δy = [Δi_out_d, Δi_out_q]

        State-space representation in tableau form:

                │   Δx  │   Δu
        ────────────────────────
        dΔx/dt  │   A   │   B
        ────────────────────────
        Δy      │   C   │   D

                    │ Δz_vc_d  Δz_vc_q  │   Δv_ref_d  Δv_ref_q   Δv_d       Δv_q        Δi_d  Δi_q  Δw
        ────────────────────────────────────────────────────────────────────────────────────────────────────────
        dΔz_vc_d/dt │  0       0        │   ki         0         -ki        0           0     0     0
        dΔz_vc_q/dt │  0       0        │   0          ki        0          -ki         0     0     0     
        ────────────────────────────────────────────────────────────────────────────────────────────────────────
        Δi_out_d    │  1       0        │   kp         0         -kp        -cf*(w)ₒ    kff   0     -cf * (v_q)ₒ
        Δi_out_q    │  0       1        │   0          kp        cf*(w)ₒ    -kp         0     kff   cf * (v_d)ₒ

        """

        kp, ki, kff, cf = self.kp_pu, self.ki_puHz, self.kffi, self.cf_pu

        A = np.zeros((2, 2))
        B = np.array([
            [ki, 0, -ki, 0, 0, 0, 0],
            [0, ki, 0, -ki, 0, 0, 0]
        ])
        C = np.eye(2)
        D = np.array([
            [kp, 0, -kp, -cf * w, kff, 0, -cf * v_q],
            [0, kp, cf * w, -kp, 0, kff, cf * v_d]
        ])

        x = DynamicalVariables(
            name=['z_vc_d', 'z_vc_q'],
            init=[z_vc_d, z_vc_q]
        )
        u = DynamicalVariables(
            name=['v_ref_d', 'v_ref_q', 'v_d', 'v_q', 'i_d', 'i_q', 'w']
        )

        i_out_d = z_vc_d + self.kffi * i_d - self.cf_pu * v_q * w # Output current of the inner voltage controller in d-axis
        i_out_q = z_vc_q + self.kffi * i_q + self.cf_pu * v_d * w # Output current of the inner voltage controller in q-axis

        y = DynamicalVariables(
            name=['i_out_d', 'i_out_q'],
            init=[i_out_d, i_out_q]
        )

        return StateSpaceModel(
            A = A,
            B = B,
            C = C,
            D = D,
            x = x,
            u = u,
            y = y
        )
