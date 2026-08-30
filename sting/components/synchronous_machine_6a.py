# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
from scipy.linalg import inv
import copy

# ------------------
# Import sting code
# ------------------
from sting.components.inner_current_controller_2a import InitialConditionsEMT
from sting.generator.core import Generator
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables, QuadraticBilinearModel
from sting.modules.simulation_emt.utils import VariablesEMT

from sting.utils.transformations import R_DQ2dq, R_dq2DQ, d_DQ2dq_dangle, d_dq2DQ_dangle

# ------------------
# Import sting code
# ------------------
class InitialConditionsEMT(NamedTuple):
    angle: float
    i_d: float
    i_q: float
    i_0: float
    i_fd: float
    i_1d: float
    i_1q: float
    i_2q: float
    v_fd: float
    v_d: float
    v_q: float
    v_0: float
    v_a: float
    v_b: float
    v_c: float
    i_D: float
    i_Q: float


@dataclass(slots=True, kw_only=True, eq=False)
class SynchronousMachine6A:
    x_d_pu: float 
    x_q_pu: float 
    x_0_pu: float
    x_ad_pu: float
    x_aq_pu: float
    x_ffd_pu: float
    x_f1d_pu: float
    x_11d_pu: float
    x_11q_pu: float
    r_a_pu: float
    r_fd_pu: float
    r_1d_pu: float
    r_1q_pu: float
    w_base: float

    A: np.ndarray = None
    B: np.ndarray = None
    N: np.ndarray = None
    L: np.ndarray = None

    def __post_init__(self):
        self._compute_dynamics_matrices()

    @classmethod
    def from_standard_parameters(cls,   x_d_pu, x_q_pu, x_l_pu, x_0_pu,
                                        x_td_pu, x_tq_pu, 
                                        x_std_pu, x_stq_pu,
                                        t_td0_s, t_tq0_s, t_std0_s, t_stq0_s,
                                        r_a_pu,
                                        w_base,
                                        x_f1d_pu=None):
        """Compute the machine fundamental parameters from the standard parameters"""
        # Compute unsaturated reactances
        x_ad = x_d_pu - x_l_pu
        x_aq = x_q_pu - x_l_pu

        # Compute rotor leakage reactances
        x_fd = ( 1/(x_td_pu - x_l_pu) - 1/(x_ad) )**(-1)  
        x_1q = ( 1/(x_tq_pu - x_l_pu) - 1/(x_aq) )**(-1)

        x_1d = ( 1/(x_std_pu - x_l_pu) - 1/(x_fd) - 1/(x_ad) )**(-1)
        x_2q = ( 1/(x_stq_pu - x_l_pu) - 1/(x_1q) - 1/(x_aq) )**(-1)

        # Compute rotor resistances
        r_fd = 1/(t_td0_s * w_base) * (x_ad + x_fd)
        r_1d = 1/(t_std0_s * w_base) * (x_1d + (x_ad * x_fd)/(x_ad + x_fd))

        r_1q = 1/(t_tq0_s * w_base) * (x_aq + x_1q)
        r_2q = 1/(t_stq0_s * w_base) * (x_2q + (x_aq * x_1q)/(x_aq + x_1q))

        # The field to damper inductance, l_f1d, is typically assumed to 
        # be equal to the inductance l_ad
        if x_f1d_pu is None:
            x_f1d_pu = x_ad

        return cls(
                x_d_pu=x_d_pu, x_q_pu=x_q_pu, x_0_pu=x_0_pu,
                x_ad_pu=x_ad, x_aq_pu=x_aq,
                x_fd_pu=x_fd, x_1d_pu=x_1d, x_1q_pu=x_1q, x_2q_pu=x_2q,
                r_fd_pu=r_fd, r_1d_pu=r_1d, r_1q_pu=r_1q, r_2q_pu=r_2q,
                w_base=w_base,
                x_f1d_pu=x_f1d_pu
                )
        
    def _compute_dynamics_matrices(self):

        # Define the inductance matrix
        l_d = self.x_d_pu
        l_q = self.x_q_pu
        l_0 = self.x_0_pu
        l_ad = self.x_ad_pu
        l_aq = self.x_aq_pu

        l_ffd = self.x_ffd_pu
        l_f1d = self.x_f1d_pu
        l_11d = self.x_11d_pu
        l_11q = self.x_11q_pu          

        r_a = self.r_a_pu
        r_fd = self.r_fd_pu
        r_1d = self.r_1d_pu
        r_1q = self.r_1q_pu

        L = np.array([
        #     i_0   i_d   i_q     i_f   i_1d     i_1q   
            [ l_0,    0,    0,     0,      0,    0   ], # λ_0
            [    0, l_d,    0,  l_ad,   l_ad,    0   ], # λ_d
            [    0,    0, l_q,     0,      0,    l_aq], # λ_q
            [    0, l_ad,    0,l_ffd,  l_f1d,    0   ], # λ_fd
            [    0, l_ad,    0,l_f1d,  l_11d,    0   ], # λ_1d
            [    0,    0, l_aq,    0,      0,    l_11q], # λ_1q
        ])
        invL = inv(L)
        self.L = L

        R = np.diag([r_a, r_a, r_a, r_fd, r_1d, r_1q])

        # Frequency coupling
        T_w = np.zeros((6, 6))
        T_w[1, 2] = +1
        T_w[2, 1] = -1

        self.A = -self.w_base * (invL @ R)
        self.B = -self.w_base * (invL)[:,:4] # Damper input voltages = 0
        self.N = -self.w_base * (invL @ T_w @ L)

    def define_variables_emt_abc(self):
        # States 
        x = DynamicalVariables(
            name = ["i_0", "i_d", "i_q", "i_fd", "i_1d", "i_1q"],
            component = f"{self.type_}_{self.id}",
            init = [self.emt_init.i_0,
                    self.emt_init.i_d,
                    self.emt_init.i_q,
                    self.emt_init.i_fd,
                    self.emt_init.i_1d,
                    self.emt_init.i_1q]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["v_fd", "w", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[  self.emt_init.v_fd,
                    self.w_base,
                    self.emt_init.v_bus_a,
                    self.emt_init.v_bus_b,
                    self.emt_init.v_bus_c]
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[self.emt_init.i_bus_a,
                  self.emt_init.i_bus_b,
                  self.emt_init.i_bus_c]
        )

    def get_derivatives_step_emt_dq0(self, i_0, i_d, i_q, i_fd, i_1d, i_1q, v_0, v_d, v_q, v_fd, w):

        i = np.array([i_0, i_d, i_q, i_fd, i_1d, i_1q])
        v = np.array([v_0, v_d, v_q, -v_fd])
        di_dt = self.A@i + self.B@v + w*self.N@i

        return di_dt