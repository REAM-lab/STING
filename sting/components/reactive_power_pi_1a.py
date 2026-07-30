from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel


class InitialConditionsEMT(NamedTuple):
    """Store the initial conditions of the reactive power controller for the EMT simulation."""
    q_ref: float
    i_ref_q: float
    z_rpc: float


@dataclass(slots=True)
class ReactivePowerPI1A:
    kp_pu: float
    ki_puHz: float
    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, q_ref: float, i_ref_q: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Parameters
        ----------
        - q_ref [pu]: Steady-state reactive power
        - i_ref_q [pu]: Steady-state d-axis current reference

        Returns
        -------
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            q_ref = q_ref,
            i_ref_q = i_ref_q,
            z_rpc = i_ref_q
        )

        return self.emt_init

    def get_small_signal_model(self, z_rpc, q_ref, i_d, i_q, v_d, v_q):
        """
        Returns the small-signal state-space model of the reactive power controller.

        Parameters
        ----------
        - z_rpc: initial condition of the PI controller integrator state.
        - q_ref [pu]: initial condition of the reference reactive power.
        - i_d [pu]: initial condition of d-axis current---e.g., the current going to the point of common coupling.
        - i_q [pu]: initial condition of q-axis current---e.g., the current going to the point of common coupling.
        - v_d [pu]: initial condition of d-axis voltage---e.g., the voltage of the LCL filter shunt OR point of common coupling.
        - v_q [pu]: initial condition of q-axis voltage---e.g., the voltage of the LCL filter shunt OR point of common coupling.

        Returns
        -------
        - ssm: Small signal model
        

        Small-signal dynamics
        ---------------------
        The dynamics of the reactive power controller are governed by:
            dΔz/dt = k_i * (Δq_ref - Δq)
            Δi_ref_q = z + k_p (Δq_ref - Δq)

        where Δq is the linearized reactive power measurement given by:
            Δq = (v_q)ₒ * Δi_d  - (v_d)ₒ * Δi_q - (i_q)ₒ * Δv_d + (i_d)ₒ * Δv_q
            
        State-space model
        -----------------
        State vector, input vector, and output vector are:
            Δx = [Δz]
            Δu = [Δq_ref, Δi_d, Δi_q, Δv_d, Δv_q]
            Δy = [Δi_ref_q]

        while the matrix representation in tableau form is:
                   │  Δz │  Δq_ref     Δi_d            Δi_q            Δv_d             Δv_q
            ───────┼─────┼──────────────────────────────────────────────────────────────────────
            dΔz/dt │  0  │  -k_i    k_i*(v_q)ₒ      -k_i*(v_d)ₒ     -k_i*(i_q)ₒ      k_i*(i_d)ₒ 
            ───────┼─────┼───────────────────────────────────────────────────────────────────────
            Δϕ     │  1  │  -k_p    k_p*(v_q)ₒ      -k_p*(v_d)ₒ     -k_p*(i_q)ₒ      k_p*(i_d)ₒ 
        """
        ssm = StateSpaceModel(
            A = np.array([[0]]),
            B = -self.ki_puHz * np.array([[1, -v_q, v_d, i_q, -i_d]]),
            C = np.array([[1]]),
            D = -self.kp_pu * np.array([[1, -v_q, v_d, i_q, -i_d]]),
            x = DynamicalVariables(name=['z_rpc'], init=z_rpc),
            u = DynamicalVariables(name=["q_ref", "i_d", "i_q", "v_d", "v_q"], init = [q_ref, i_d, i_q, v_d, v_q]),
            y = DynamicalVariables(name=['i_ref_q'])
        )
        return ssm

    def define_variables_emt_abc(self):

        # States 
        x = DynamicalVariables(
            name = ['z_rpc'],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.z_rpc]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["q_ref", "q"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.q_ref, self.emt_init.q_ref]
        )

        return [x, u]

    def get_derivatives_step_emt_abc(self, q_ref: float, q: float, z_rpc: float) -> float:
        """
        Returns the derivatives of the states of the reactive power controller for the EMT simulation step.
        
        Parameters
        ----------
        - q_ref [pu]: Reactive power reference
        - q [pu]: Reactive power
        - z_rpc [pu]: State variable associated to the PI controller
        
        Returns
        -------
        - d_z_rpc [pu/s]: Derivative of the state variable associated to the PI controller
        """

        # Compute derivative of the state variable associated to the PI controller
        d_z_rpc = self.ki_puHz * (-1) * (q_ref - q)

        return d_z_rpc

    def get_algebraics_step_emt_abc(self, q_ref: float, q: float, z_rpc: float) -> list[float]:
        """
        Returns the algebraic outputs of the reactive power controller for the EMT simulation step.

        Parameters
        ----------
        - q_ref [pu]: Reactive power reference
        - q [pu]: Reactive power
        - z_rpc [pu]: State variable associated to the PI controller

        Returns
        -------
        - i_ref_q [pu]: q-axis current reference
        """

        # Compute q-axis current reference
        i_ref_q = self.kp_pu * (-1) * (q_ref - q) + z_rpc

        return i_ref_q
