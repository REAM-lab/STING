from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel


class InitialConditionsEMT(NamedTuple):
    """Store the initial conditions of the active power controller for the EMT simulation."""
    p_ref: float
    i_ref_d: float
    z_apc: float


@dataclass(slots=True)
class ActivePowerPI1A:
    kp_pu: float
    ki_puHz: float
    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, p_ref: float, i_ref_d: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Parameters
        ----------
        - p_ref [pu]: Steady-state active power
        - i_ref_d [pu]: Steady-state d-axis current reference
         
        Returns
        -------
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            p_ref = p_ref,
            i_ref_d = i_ref_d,
            z_apc = i_ref_d
        )

        return self.emt_init

    def get_small_signal_model(self, z_apc, p_ref, i_d, i_q, v_d, v_q):
        """
        Returns the small-signal state-space model of the active power controller.

        Parameters
        ----------
        - z_apc: initial condition of the PI controller integrator state.
        - p_ref [pu]: initial condition of the reference active power.
        - i_d [pu]: initial condition of d-axis current---e.g., the current going to the point of common coupling.
        - i_q [pu]: initial condition of q-axis current---e.g., the current going to the point of common coupling.
        - v_d [pu]: initial condition of d-axis voltage---e.g., the voltage of the LCL filter shunt OR point of common coupling.
        - v_q [pu]: initial condition of q-axis voltage---e.g., the voltage of the LCL filter shunt OR point of common coupling.
        
        Returns
        -------
        - ssm: Small signal model
        
        
        Small-signal dynamics
        ---------------------
        The dynamics of the active power controller are governed by:
            dΔz/dt = k_i * (Δp_ref - Δp)
            Δi_ref_d = z + k_p (Δp_ref - Δp)

        where Δp is the linearized active power measurement given by:
            Δp = (v_d)ₒ * Δi_d + (v_q)ₒ * Δi_q + (i_d)ₒ * Δv_d + (i_q)ₒ * Δv_q

            
        State-space model
        -----------------
        State vector, input vector, and output vector are:
            Δx = [Δz]
            Δu = [Δp_ref, Δi_d, Δi_q, Δv_d, Δv_q]
            Δy = [Δi_ref_d]

        while the matrix representation in tableau form is:
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
            x = DynamicalVariables(name=['z_apc'], init=z_apc),
            u = DynamicalVariables(name=["p_ref", "i_d", "i_q", "v_d", "v_q"], init = [p_ref, i_d, i_q, v_d, v_q]),
            y = DynamicalVariables(name=['i_ref_d'])
        )
        return ssm

    def get_quadratic_bilinear_model(self, z_apc, p_ref, p):
        """
        Quadratic bilinear dynamics
        ---------------------------
        The dynamics of the active power controller are governed by:
            dz/dt = k_i * (p_ref - p)
            i_ref_d = z + k_p * (p_ref - p)        
        """
        ssm = StateSpaceModel(
            A = np.array([[0]]),
            B = self.ki_puHz * np.array([[1, -1]]),
            C = np.array([[1]]),
            D = self.kp_pu * np.array([[1, -1]]),
            x = DynamicalVariables(name=['z_apc'], init=z_apc),
            u = DynamicalVariables(name=["p_ref", "p_apc"], init = [p_ref, p]),
            y = DynamicalVariables(name=['i_ref_d'])
        )
        return ssm.to_quadratic_bilinear()
    
    def define_variables_emt_abc(self):

        # States 
        x = DynamicalVariables(
            name = ['z_apc'],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.z_apc]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["p_ref", "p"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.p_ref, self.emt_init.p_ref]
        )

        return [x, u]

    def get_derivatives_step_emt_abc(self, p_ref: float, p: float, z_apc: float) -> float:
        """
        Returns the derivatives of the states of the active power controller for the EMT simulation step.
        
        Parameters
        ----------
        - p_ref [pu]: Active power reference
        - p [pu]: Active power
        - z_apc [pu]: State variable associated to the PI controller
        
        Returns
        -------
        - d_z_apc [pu/s]: Derivative of the state variable associated to the PI controller
        """

        # Compute derivative of the state variable associated to the PI controller
        d_z_apc = self.ki_puHz * (p_ref - p)

        return d_z_apc

    def get_algebraics_step_emt_abc(self, p_ref: float, p: float, z_apc: float) -> list[float]:
        """
        Returns the algebraic outputs of the active power controller for the EMT simulation step.

        Parameters
        ----------
        - p_ref [pu]: Active power reference
        - p [pu]: Active power
        - z_apc [pu]: State variable associated to the PI controller

        Returns
        -------
        - i_ref_d [pu]: d-axis current reference
        """

        # Compute d-axis current reference
        i_ref_d = self.kp_pu * (p_ref - p) + z_apc

        return i_ref_d
