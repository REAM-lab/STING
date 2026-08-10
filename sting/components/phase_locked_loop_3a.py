from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from sting.utils.dynamical_systems import DynamicalVariables, QuadraticBilinearModel, StateSpaceModel
from sting.utils.transformations import abc2dq0, dq02abc


class InitialConditionsEMT(NamedTuple):
    theta_pll: float
    v_pll_q: float
    z_pll: float
    v_a: float
    v_b: float
    v_c: float

@dataclass
class PhaseLockedLoop3A:
    """
    A third-order model of a phase-locked loop with a filter.  
                                                              w_base                           
                                                                 │
               ┌─────────┐     ┌─────────────┐     ┌────┐    [+] ▼[+]  ┌──────┐
    v_abc ────▶│ abc→dq0 │────▶│ 1/(tau*s+1) │────▶│ PI │─────▶──┴────▶│ wb/s │───┬──▶ θ_pll
               └───┬─────┘     └─────────────┘ v_q └────┘    Δw        └──────┘   │
                   ▲                                                              │
                   └──────────────────────────────────────────────────────────────┘

    Parameters
    - kp_pu: Proportional gain [pu]
    - ki_puHz: Integral gain [pu]
    - tau: Filter constant [pu]
    - wbase: Nominal frequency [rad/s] of the system
    - alpha: Quadratic bilinear artificial stabilization
    """
    kp_rad_s: float
    ki_rad2_s2: float
    tau: float
    wbase: float
    alpha: float = 0


    def get_steady_state(self, v_mag, relative_phase_deg):

        theta_pll = relative_phase_deg * np.pi / 180
        v_a, v_b, v_c = dq02abc(v_mag, 0, 0, theta_pll)

        self.emt_init = InitialConditionsEMT(
            theta_pll = theta_pll,
            v_pll_q = 0.0,
            z_pll = 0.0,
            v_a = v_a,
            v_b = v_b,
            v_c = v_c
        )

    
    def get_small_signal_model(self, v_mag, relative_phase_deg):
        
        phase_rad = relative_phase_deg*np.pi/180
        wb = self.wbase
        sin0 = np.sin(phase_rad)
        cos0 = np.cos(phase_rad)
        
        ki, kp, wb, tau = self.ki_rad2_s2, self.kp_rad_s, self.wbase, self.tau
        
        A = np.array([
            [-1/tau, 0, -v_mag/tau], # v_filter_q
            [    ki, 0, 0         ], # z_pll
            [    kp, 1, 0         ], # phase
           
        ])
        B = np.array([
            [-sin0/tau, cos0/tau],
            [        0,        0],
            [        0,        0],
            
        ])
        C = np.array([
            [kp/wb, 1/wb, 0], # w
            [    0,    0, 1], # phase
        ])
        D = np.zeros((2, 2))

        ssm = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            u = DynamicalVariables(name=['v_bus_D', 'v_bus_Q']),
            y = DynamicalVariables(name=['w', 'phase']),
            x = DynamicalVariables(
                name=["v_pll_q", "z_pll", "phase_pll"], 
                init=[0, 0, phase_rad] 
                )
            )
        return ssm


    def get_quadratic_bilinear_model(self, v_mag, relative_phase_deg):
        """
        The quadratic bilinear dynamics of the PLL are given by:
            d/dt z_pi = ki * v_q
            d/dt v_q  = (1/tau) * (-v_D*z_s + v_Q*z_c - v_q)
            d/dt z_s  = z_c * (w - wb)
                      =  z_c * (kp*v_q + z_pi) - z_c * wb - alpha * (z_c^2 + z_s^2 - 1)
            d/dt z_c  = -z_s * (kp*v_q + z_pi) + z_s * wb - alpha * (z_c^2 + z_s^2 - 1)

        Note: The output angular velocity is in per unit, that is w = wb * w_pu. 
            w_pu = 1/wb * (kp * v_q + z_pi)
        """

        phase_rad = relative_phase_deg*np.pi/180
        v_bus_DQ = v_mag * np.exp(phase_rad * 1j)

        ki, kp, wb, tau = self.ki_rad2_s2, self.kp_rad_s, self.wbase, self.tau
        a = self.alpha

        A = np.array([
            [-1/tau, 0, 0,  0], # v_filter_q
            [    ki, 0, 0,  0], # z_pll
            [     0, 0, 0,-wb], # z_sin
            [     0, 0,wb,  0], # z_cos
        ])
        B = np.zeros((4, 2))

        # Nonlinear dynamics of sin and cos "lifted" states
        H0 = np.zeros((4,4))
        H_sin = np.array([
            [  0,  0, 0, 0],
            [  0,  0, 0, 0],
            [  0,  0,-a, 0],
            [-kp, -1,-a, 0],
        ])
        H_cos = np.array([
            [  0,  0, 0, 0],
            [  0,  0, 0, 0],
            [ kp,  1, 0,-a],
            [  0,  0, 0,-a],
        ])
        H = np.hstack([H0, H0, H_sin, H_cos])

        # Inputs-state interactions of xy -> dq voltage
        # v_q = -v_D * sin + v_Q * cos 
        N_D = np.array([
            [0, 0,-1/tau, 0], # v_D * z_sin
            [0, 0,     0, 0],
            [0, 0,     0, 0],
            [0, 0,     0, 0],
        ])
        N_Q = np.array([
            [0, 0, 0,1/tau], # v_Q * z_cos
            [0, 0, 0,    0],
            [0, 0, 0,    0],
            [0, 0, 0,    0],
        ])
        N = np.hstack([N_D, N_Q])

        C = np.array([
            [kp/wb, 1/wb, 0, 0], # w
            [    0,    0, 1, 0], # z_sin
            [    0,    0, 0, 1], # z_cos
        ])

        D = np.zeros((3, 2))

        u = DynamicalVariables(
            name=['v_bus_D', 'v_bus_Q'],
            init=[v_bus_DQ.real, v_bus_DQ.imag])
        y = DynamicalVariables(name=['w', 'sin', 'cos'])
        x = DynamicalVariables(
            name=["v_pll_q", "z_pll", "sin_pll", "cos_pll"], 
            init=[0, wb, np.sin(phase_rad), np.cos(phase_rad)] 
        )

        return QuadraticBilinearModel(A=A, B=B, C=C, D=D, H=H, N=N, x=x, y=y, u=u)


    def get_derivatives_step_emt_abc(self, v_pll_q, z_pll, theta_pll: float, v_a: float, v_b: float, v_c: float):
        # Get voltage voltage of axis q
        _, v_q, _ = abc2dq0(v_a, v_b, v_c, theta_pll)
        # Voltage filter dynamics
        d_v_pll_q = (1/self.tau) * (v_q - v_pll_q)

        # Compute the derivatives of the state variables for the EMT simulation step
        d_theta_pll = (self.kp_rad_s * v_pll_q) + z_pll + self.wbase
        d_z_pll = self.ki_rad2_s2 * v_pll_q

        return [d_v_pll_q, d_z_pll, d_theta_pll]

    def get_derivatives_step_emt_dq0(self, v_pll_q, z_pll, phase_pll, v_bus_D, v_bus_Q):

        v_bus_q = -v_bus_D * np.sin(phase_pll) + v_bus_Q * np.cos(phase_pll)
        # PLL dynamics
        d_phase_pll = (self.kp_rad_s * v_pll_q) + z_pll
        d_z_pll = self.ki_rad2_s2 * v_pll_q
        # Voltage filter dynamics
        d_v_pll_q = (1/self.tau) * (v_bus_q - v_pll_q)

        return [d_v_pll_q, d_z_pll, d_phase_pll]