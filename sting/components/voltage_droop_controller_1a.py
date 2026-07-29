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

    q_ref: float
    v_ref: float

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

    def get_steady_state(self, q_ref: float, v_ref: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Inputs:
        - q [pu]: Steady-state reactive power
        - v_ref [pu]: Steady-state reference voltage
         
        Outputs:
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            q_ref = q_ref,
            v_ref = v_ref
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

        return [d_q_f]

    def get_algebraics_step_emt_dq0(self, v_ref: float, q_ref: float, q_f: float) -> list[float]:
        """
        Returns the algebraic outputs of the voltage droop control for the EMT simulation step.
        """

        # Compute reference voltage in d_axis
        v_d_ref = v_ref + self.k_q_pu * (q_ref - q_f)

        # Fix the reference voltage in q_axis to zero
        v_q_ref = 0.0

        return [v_d_ref, v_q_ref]

    def get_small_signal_model(self, i_d, i_q, v_d, v_q, q_ref, v_ref):
        """
        Returns the small-signal model of the voltage droop control.

        Inputs:
        - i_d [pu]: initial d-axis current. For example, it can be the current going to the point of common coupling (PCC).
        - i_q [pu]: initial q-axis current. For example, it can be the current going to the point of common coupling (PCC).
        - v_d [pu]: initial d-axis voltage. For example, it can be the voltage of the shunt of the LCL filter.
        - v_q [pu]: initial q-axis voltage. For example, it can be the voltage of the shunt of the LCL filter.
        - q_ref [pu]: initial reference reactive power.
        - v_ref [pu]: initial reference voltage.

        Outputs:
        - ssm: State-space model object.

        Equations to derive the small-signal model:
        dΔq_f/dt = w_q_puHz * (Δq - Δq_f)
                 = w_q_puHz * ( -(v_d)ₒ * Δi_q + (v_q)ₒ * Δi_d + (i_d)ₒ * Δv_q - (i_q)ₒ * Δv_d - Δq_f)
        dΔv_d_ref/dt = Δv_ref + k_q_pu * (Δq_ref - Δq_f)

        State vector, input vector, and output vector are:
        Δx = [Δq_f]
        Δu = [Δq_ref, Δv_ref, Δi_d, Δi_q, Δv_d, Δv_q]
        Δy = [Δv_d_ref, Δv_q_ref]

        where:
        - q_f: State variable associated to low-pass filter
        - q_ref: Reference reactive power
        - v_ref: Reference voltage
        - i_d: d-axis current
        - i_q: q-axis current
        - v_d: d-axis voltage
        - v_q: q-axis voltage  

        State-space representation in tableau form:
        
                │   Δx  │   Δu
        ────────────────────────
        dΔx/dt  │   A   │   B 
        ────────────────────────
        Δy      │   C   │   D

                    │ Δq_f      │   Δq_ref    Δv_ref    Δi_d            Δi_q              Δv_d            Δv_q
        ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        dΔq_f/dt    │ -w_q      │   0         0         w_q * (v_q)ₒ    -w_q * (v_d)ₒ     -w_q * (i_q)ₒ    w_q * (i_d)ₒ
        ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Δv_d_ref    │ 0         │   k_q       1         0               0                 0                0
        Δv_q_ref    │ 0         │   0         0         0               0                 0                0
        """

        w_q = self.w_q_puHz
        k_q = self.k_q_pu

        A = np.array([
            [-w_q]
        ])
        B = np.array([
            [0, 0, w_q * v_q, -w_q * v_d, -w_q * i_q, w_q * i_d]
        ])
        C = np.array([
            [0],
            [0]
        ])
        D = np.array([
            [k_q, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        ssm = StateSpaceModel(
            A = A,
            B = B,
            C = C,
            D = D,
            x = DynamicalVariables(
                name=["q_f"],
                init=[q_ref]
            ),
            u = DynamicalVariables(name=["q_ref", "v_ref", "i_d", "i_q", "v_d", "v_q"],
                                   init=[q_ref, v_ref, i_d, i_q, v_d, v_q]),
            y = DynamicalVariables( name=["v_d_ref", "v_q_ref"],
                                    init=[v_ref, 0])
        )
        return ssm