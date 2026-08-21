import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.utils.dynamical_systems import (
    DynamicalVariables,
    QuadraticBilinearModel,
    StateSpaceModel,
)


class InitialConditionsEMT(NamedTuple):
    angle: float
    w: float
    p_ref: float


@dataclass(slots=True)
class VirtualInertia2A:
    """
    The virtual inertia model is a second-order model that emulates the dynamics of a synchronous generator.

    Parameters:
    - kd_w_pu: damping gain [pu] of the active power controller
    - h_s: virtual inertia [s]
    - w_nom: nominal frequency [rad/s] of the system

    """
    h_s: float
    kd_w_pu: float
    w_nom: float
    alpha: float = 0

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, angle: float, w: float, p_ref: float) -> InitialConditionsEMT:
        """
        Returns the initial conditions for the EMT simulation based on the steady-state values of the system.
        
        Inputs:
        - angle [rad]: Steady-state angle of the active power controller
         
        Outputs:
        - emt_init: Initial conditions for the EMT simulation
        """

        self.emt_init = InitialConditionsEMT(
            angle = angle,
            w = w,
            p_ref = p_ref
        )

        return self.emt_init

    def get_derivatives_step_emt_abc(self, w: float, p_ref: float, p: float) -> list[float]:
        """
        Compute the derivates with respect to time of the states of the virtual inertia model
        for the next time step in the EMT simulation.

        Inputs:
        - w [pu]: angular frequency. It is a state of the virtual inertia model.
        - p_ref [pu]: reference active power. It is an input to the virtual inertia model.
        - p [pu]: measured active power. It is an input to the virtual inertia model.

        Outputs:
        - d_angle [rad/s]: derivative of the angle of the active power controller
        - d_w [pu/s]: derivative of the angular frequency of the active power controller
        """
    
        # Extract the list of parameters
        kd_w = self.kd_w_pu  # damping gain of active power controller
        h = self.h_s  # virtual inertia
        w_nom = self.w_nom  # nominal frequency of the system

        # Derivative of the angle
        d_angle_pc = w_nom * w
        
        # Derivative of the angular frequency
        d_w_pc = 1/(2 * h) * (p_ref - p - kd_w * (w - 1))
    
        return [d_angle_pc, d_w_pc]

    def get_quadratic_bilinear_model(self, w:float, angle_rad:float, p_ref:float, p:float):
        """
        The contents of this function should not be presented as original work by another author.

        Parameters
        ----------
        - w [pu]: Initial angular frequency in per unit (state 1).
        - angle_rad: Reference angle in radians.
        - p_ref [pu]: Initial reference active power (input 1).
        - p [pu]: Initial measured active power (input 3).

        Dynamics
        --------
        The quadratic bilinear model dynamics are given by:
            2h d/dt w  = p_ref - p - kd * (w - 1)
            d/dt z_sin = wb * z_cos * (w - 1) - alpha * (z_sin^2 + z_cos^2 - 1) 
            d/dt z_cos =wb* -z_sin * (w - 1) - alpha * (z_sin^2 + z_cos^2 - 1) 

        States, inputs and outputs:
            x = [w, sin, cos]
            u = [p_ref, one, p]
            y = x
        Note that the second input is a dummy variable equal to 1 for all time.
        """

        h, kd, wb, a = self.h_s, self.kd_w_pu, self.w_nom, self.alpha

        A = np.array([
            [-kd/(2*h),  0,  0], # w_pu
            [        0,  0,-wb], # sin
            [        0, wb,  0], # cos
        ])

        B = np.array([
        #   | p_ref |  u_one  |    p    |
            [1/(2*h), kd/(2*h), -1/(2*h)],
            [      0,        a,        0],
            [      0,        a,        0], 
        ])

        H_sin = np.array([
        #   w*s | s^2 | c*s 
            [  0,   0,   0],
            [  0,  -a,   0],
            [-wb,  -a,   0]
        ])

        H_cos = np.array([
        #   w*c | s*c | c^2 
            [ 0,   0,   0],
            [wb,   0,  -a],
            [ 0,   0,  -a]
        ])

        H = np.hstack((np.zeros((3,3)), H_sin, H_cos))

        x = DynamicalVariables(name=["w", "sin", "cos"], init=[w, np.sin(angle_rad), np.cos(angle_rad)])
        u = DynamicalVariables(name=["p_ref", "one" ,"p"], init=[p_ref, 1, p])
        y = copy.deepcopy(x)

        return QuadraticBilinearModel(A=A, B=B, C=np.eye(3), D=np.zeros((3,3)), H=H, N=np.zeros((3,9)), x=x, y=y, u=u)

    def get_small_signal_model(self, i_d, i_q, v_d, v_q, angle, p_ref):
        """
        Returns the small-signal state-space model of the virtual inertia model.

        Inputs:
        - i_d [pu]: initial condition of d-axis current. For example, it can be the current going to the point of common coupling (PCC).
        - i_q [pu]: initial condition of q-axis current. For example, it can be the current going to the point of common coupling (PCC).
        - v_d [pu]: initial condition of d-axis voltage. For example, it can be the voltage of the shunt of the LCL filter.
        - v_q [pu]: initial condition of q-axis voltage. For example, it can be the voltage of the shunt of the LCL filter.
        - angle [rad]: initial condition of the angle of the active power controller.
        - p_ref [pu]: initial condition of the reference active power.

        Equations to derive the small-signal model:
        dΔϕ/dt = ω_nom * Δω
        dΔω/dt = 1/(2 * h) * (Δp_ref - Δp - kd_w * Δω)
               = 1/(2 * h) * (Δp_ref - (v_d)ₒ * Δi_d - (v_q)ₒ * Δi_q - (i_d)ₒ * Δv_d - (i_q)ₒ * Δv_q - kd_w * Δω)

        where:
        - ϕ: angle of the active power controller relative to synchronous reference frame
        - ω: angular frequency of the active power controller
        - p_ref: reference active power
        - p: measured active power
        - i_d: d-axis current
        - i_q: q-axis current
        - v_d: d-axis voltage
        - v_q: q-axis voltage

        State vector, input vector, and output vector are:
        Δx = [Δϕ, Δω]
        Δu = [Δp_ref, Δi_d, Δi_q, Δv_d, Δv_q]
        Δy = [Δϕ, Δω]

        State-space representation in tableau form:

                │   Δx  │   Δu
        ────────────────────────
        dΔx/dt  │   A   │   B 
        ────────────────────────
        Δy      │   C   │   D

                │ Δϕ  Δω           │   Δp_ref    Δi_d            Δi_q            Δv_d             Δv_q
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        dΔϕ/dt │ 0   ω_nom         │   0         0                0               0               0
        dΔω/dt │ 0   -kd_w/(2*h)   │   1/(2*h)   -(v_d)ₒ/(2*h)    -(v_q)ₒ/(2*h)   -(i_d)ₒ/(2*h)   -(i_q)ₒ/(2*h)
        ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        Δϕ     │ 1   0             │   0         0                0               0               0
        Δω     │ 0   1             │   0         0                0               0               0
        """ 

        h, kd_w, w_nom = self.h_s, self.kd_w_pu, self.w_nom

        A = np.array([
                        [0, w_nom],     
                        [0, -kd_w/(2*h)]
                    ])

        B = np.array([
                        [0, 0, 0, 0, 0],      
                        [1/(2*h), -(v_d)/(2*h), -(v_q)/(2*h), -(i_d)/(2*h), -(i_q)/(2*h)]
                    ])

        C = np.eye(2)

        D = np.zeros((2, 5))


        ssm = StateSpaceModel(
            A = A,
            B = B,
            C = C,
            D = D,
            x = DynamicalVariables(
                name=["angle", "w"],
                init=[angle, 1]
            ),
            u = DynamicalVariables(name=["p_ref", "i_d", "i_q", "v_d", "v_q"],
                                   init=[p_ref, i_d, i_q, v_d, v_q]),
            y = DynamicalVariables( name=["angle", "w"],
                                    init=[angle, 1])
        )
        return ssm