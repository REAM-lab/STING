from dataclasses import dataclass
from typing import NamedTuple
import numpy as np

from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

class InitialConditionsEMT(NamedTuple):
    pass

@dataclass
class PhaseLockedLoop3A:
    """
    A third-order model of a phase-locked loop with a filter.
    """
    kp_pu: float
    ki_puHz: float
    tau: float
    wbase: float
    alpha: float = 0


    def get_steady_state(self):
        pass

    
    def small_signal_model(self, v_bus_mag, relative_phase_deg):
        
        v_mag, phase_rad = v_bus_mag, (relative_phase_deg*np.pi/180)
        wb = self.wbase
        sin0 = np.sin(phase_rad)
        cos0 = np.cos(phase_rad)
        
        ki, kp, wb, tau = self.ki_puHz, self.kp_pu, self.wbase, self.tau
        
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
        D = np.zeros((2, 3))

        ssm = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            u = DynamicalVariables(name=['v_bus_D', 'v_bus_Q']),
            y = DynamicalVariables(name=['phase', 'w']),
            x = DynamicalVariables(
                name=["z_pll", "phase_pll", "v_pll_q"], 
                init=[0, phase_rad, 0] 
                )
            )
        return ssm


    def quadratic_bilinear_model(self, v_bus_mag, relative_phase_deg):
        
        ki, kp, wb, tau = self.ki_puHz, self.kp_pu, self.wbase, self.tau
        a = self.alpha

        A = np.array([
            [-1/tau, 0, 0, 0], # v_filter_q
            [    ki, 0, 0, 0], # z_pll
            [     0, 0, 0, 0], # z_sin
            [     0, 0, 0, 0], # z_cos
        ])
        B = np.zeros((4, 2))

        # Nonlinear dynamics of sin and cos "lifted" states
        H0 = np.zeros((5,5))
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

        # Inputs-state interactions of DQ -> dq voltage
        # v_d = -v_D * sin + v_Q * cos 
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

        D = np.zeros((2, 4))

        return 


    def differential_step_emt_dq0(self, z_pll, v_pll_q, v_bus_q):
        # PLL dynamics
        d_theta_pll = (self.kp_pu * v_pll_q) + z_pll + self.wbase
        d_z_pll = self.ki_puHz * v_pll_q
        # Voltage filter dynamics
        d_v_pll_q = (1/self.tau) * (v_bus_q - v_pll_q)

        return [d_theta_pll, d_z_pll, d_v_pll_q]