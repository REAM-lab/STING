# ---------------------------------------
# Import libraries
# ---------------------------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple

# ---------------------------------------
# Import sting code
# ---------------------------------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.utils.transformations import dq02abc, abc2dq0


# ---------------------------------------
# Sub-classes
# ---------------------------------------
class InitialConditionsEMT(NamedTuple):
    theta_pll: float
    z_pll: float
    v_a: float
    v_b: float
    v_c: float

# ---------------------------------------
# Main class
# ---------------------------------------
@dataclass(slots=True)
class PhaseLockedLoop2A:
    kp_pu: float
    ki_puHz: float
    wbase: float

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, v_bus_mag, relative_phase_deg):

        theta_pll = relative_phase_deg * np.pi / 180
        v_a, v_b, v_c = dq02abc(v_bus_mag, 0, 0, theta_pll)

        self.emt_init = InitialConditionsEMT(
            theta_pll = theta_pll,
            z_pll = 0.0,
            v_a = v_a,
            v_b = v_b,
            v_c = v_c
        )

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

    def define_variables_emt_abc(self):

        # States 
        x = DynamicalVariables(
            name = ['theta_pll', 'z_pll'],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.theta_pll, self.emt_init.z_pll]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["v_a", "v_b", "v_c"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.v_a, self.emt_init.v_b, self.emt_init.v_c]
        )

        return [x, u]
    
    def get_derivatives_step_emt_abc(self, theta_pll: float, z_pll: float, v_a: float, v_b: float, v_c: float):
        """
        Returns a step of differential equations that describe the PLL dynamics.
        The PLL tracks the phase of the grid voltage.

        Inputs:
        - theta_pll [rad]: State variable associated to the phase of the PLL
        - z_pll [pu]: State variable associated to integral control block of the PLL
        - v_a [pu]: Phase A voltage of the grid
        - v_b [pu]: Phase B voltage of the grid
        - v_c [pu]: Phase C voltage of the grid

        Outputs:
        - d_theta_pll: Derivative of the state associated to the phase of the PLL
        - d_z_pll: Derivative of the state associated to integral control block of the PLL
        """

        # Get voltage voltage of axis q
        _, v_q, _ = abc2dq0(v_a, v_b, v_c, theta_pll)

        # Compute the derivatives of the state variables for the EMT simulation step
        d_theta_pll = (self.kp_pu * v_q) + z_pll + self.wbase
        d_z_pll = self.ki_puHz * v_q

        return [d_theta_pll, d_z_pll]