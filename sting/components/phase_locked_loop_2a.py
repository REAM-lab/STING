import numpy as np
from dataclasses import dataclass

from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

@dataclass(slots=True)
class PhaseLockedLoop2A:
    kp_pu: float
    ki_puHz: float
    wbase: float

    def get_steady_state(self):
        return

    def get_small_signal_model(self, v_bus_mag, relative_phase_deg):
        kp, ki = self.kp_pu, self.ki_puHz
        # Compute the reference phase angle in radians
        v_mag, phase_rad = v_bus_mag, (relative_phase_deg*np.pi/180)
        wb = self.wbase
        sin0 = np.sin(phase_rad)
        cos0 = np.cos(phase_rad)

        A = np.array([  
            [0         ,   -v_mag * ki],
            [1         , -1*v_mag * kp]
        ])
        B = np.array([  
            [-sin0 * ki    ,     +cos0 * ki],
            [-1 * kp * sin0,  1 * kp * cos0]
        ])
        C = np.array([  
            [   0  ,                   1],
            [1/wb  ,  -1/wb * v_mag * kp]
        ])
        D = np.array([ 
            [0                 ,  0               ],
            [-1/wb * kp * sin0 ,  1/wb * kp * cos0]
        ])

        ssm = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            u = DynamicalVariables(name=['v_bus_D', 'v_bus_Q']),
            y = DynamicalVariables(name=['phase', 'w']),
            x = DynamicalVariables(
                name=["z_pll", "phase_pll"], 
                init=[0, phase_rad] 
                )
            )
        return ssm

    def differential_step_emt_dq0(self, z_pll, v_bus_q):
        """
        Returns a step of differential equations that describe the PLL dynamics.
        The PLL tracks the phase of the grid voltage.
        """

        d_theta_pll = (self.kp_pu * v_bus_q) + z_pll + self.wbase
        d_z_pll = self.ki_puHz * v_bus_q

        return [d_theta_pll, d_z_pll]