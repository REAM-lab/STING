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
    """Store the initial conditions of the active power controller for the EMT simulation."""

    p_ref: float
    i_ref_d: float
    z_pi: float


# ---------------------------------------
# Main class
# ---------------------------------------
@dataclass(slots=True)
class ActivePowerPI1A:
    kp_pu: float
    ki_puHz: float
    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, p_ref: float, i_ref_d: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Inputs:
        - p_ref [pu]: Steady-state active power
        - i_ref_d [pu]: Steady-state d-axis current reference
         
        Outputs:
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            p_ref = p_ref,
            i_ref_d = i_ref_d,
            z_pi = i_ref_d
        )

        return self.emt_init

    def get_small_signal_model(self, z_pi, p_ref, i_d, i_q, v_d, v_q):
        """
        Returns the small-signal state-space model of the active power controller.

        Parameters
        ----------
        - z_pi: initial condition of the PI controller integrator state.
        - p_ref [pu]: initial condition of the reference active power.
        - i_d [pu]: initial condition of d-axis current---e.g., the current going to the point of common coupling.
        - i_q [pu]: initial condition of q-axis current---e.g., the current going to the point of common coupling.
        - v_d [pu]: initial condition of d-axis voltage---e.g., the voltage of the LCL filter shunt OR point of common coupling.
        - v_q [pu]: initial condition of q-axis voltage---e.g., the voltage of the LCL filter shunt OR point of common coupling.
        
        Notes
        -----
        Equations to derive the small-signal model:
            dΔz/dt = k_i * (Δp_ref - Δp)
            Δi_ref_d = z + k_p (Δp_ref - Δp)
        where Δp is the linearized active power given by
            Δp = (v_d)ₒ * Δi_d + (v_q)ₒ * Δi_q + (i_d)ₒ * Δv_d + (i_q)ₒ * Δv_q
        
        State vector, input vector, and output vector are:
            Δx = [Δz]
            Δu = [Δp_ref, Δi_d, Δi_q, Δv_d, Δv_q]
            Δy = [Δi_ref_d]

        State-space representation
                   │  Δz │  Δp_ref     Δi_d            Δi_q            Δv_d             Δv_q
            ───────┼─────┼──────────────────────────────────────────────────────────────────────
            dΔz/dt │  0  │   k_i   -k_i*(v_d)ₒ      -k_i*(v_q)ₒ     -k_i*(i_d)ₒ     -k_i*(i_q)ₒ 
            ───────┼─────┼───────────────────────────────────────────────────────────────────────
            Δϕ     │  1  │   k_p   -k_p*(v_d)ₒ      -k_p*(v_q)ₒ     -k_p*(i_d)ₒ     -k_p*(i_q)ₒ 
        """
        ssm = StateSpaceModel(
            A = np.array([[0]]),
            B = self.ki_puHz * np.array([[1, -v_d, -v_q, -i_d, -i_q]]),
            C = np.array([[1]]),
            D = self.kp_pu * np.array([[1, -v_d, -v_q, -i_d, -i_q]]),
            x = DynamicalVariables(name=['z_pi'], init=z_pi),
            u = DynamicalVariables(name=["p_ref", "i_d", "i_q", "v_d", "v_q"], init = [p_ref, i_d, i_q, v_d, v_q]),
            y = DynamicalVariables(name=['i_ref_d'])
        )
        return ssm
    
    def define_variables_emt_abc(self):

        # States 
        x = DynamicalVariables(
            name = ['z_pi'],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.z_pi]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["p_ref", "p"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.p_ref, self.emt_init.p_ref]
        )

        return [x, u]

    def get_derivatives_step_emt_abc(self, p_ref: float, p: float, z_pi: float) -> float:
        """
        Returns the derivatives of the states of the active power controller for the EMT simulation step.
        
        Inputs:
        - p_ref [pu]: Active power reference
        - p [pu]: Active power
        - z_pi [pu]: State variable associated to the PI controller
        
        Outputs:
        - d_z_pi [pu/s]: Derivative of the state variable associated to the PI controller
        """

        # Compute derivative of the state variable associated to the PI controller
        d_z_pi = self.ki_puHz * (p_ref - p)

        return d_z_pi

    def get_algebraics_step_emt_abc(self, p_ref: float, p: float, z_pi: float) -> list[float]:
        """
        Returns the algebraic outputs of the active power controller for the EMT simulation step.

        Inputs:
        - p_ref [pu]: Active power reference
        - p [pu]: Active power
        - z_pi [pu]: State variable associated to the PI controller

        Outputs:
        - i_ref_d [pu]: d-axis current reference
        """

        # Compute d-axis current reference
        i_ref_d = self.kp_pu * (p_ref - p) + z_pi

        return i_ref_d
