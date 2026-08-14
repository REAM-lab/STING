# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple

# ------------------
# Import sting code
# ------------------
from sting.components.inner_current_controller_2a import InitialConditionsEMT
from sting.generator.core import Generator
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables, QuadraticBilinearModel
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.transformations import dq02abc, abc2dq0

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


@dataclass(slots=True, kw_only=True, eq=False)
class SM8A(Generator):
    x_d_pu: float 
    x_q_pu: float 
    x_td_pu: float
    x_tq_pu: float
    x_std_pu: float
    x_stq_pu: float
    t_td0_s: float
    t_tq0_s: float
    t_std0_s: float
    t_stq0_s: float
    x_l_pu: float
    r_a_pu: float
    x_0_pu: float

    x_ad_pu: float = None
    x_aq_pu: float = None
    x_fd_pu: float = None
    x_1d_pu: float = None
    x_1q_pu: float = None
    x_2q_pu: float = None
    r_fd_pu: float = None
    r_1d_pu: float = None
    r_1q_pu: float = None
    r_2q_pu: float = None
    w_base: float = None
    R: np.ndarray = None
    L: np.ndarray = None
    T: np.ndarray = None
    emt_init: InitialConditionsEMT = field(init=False)


    def __post_init__(self):

        self.w_base = 2 * np.pi * self.base_frequency_Hz

        # Compute unsaturated reactances
        x_ad = self.x_d_pu - self.x_l_pu
        x_aq = self.x_q_pu - self.x_l_pu

        # Compute rotor leakage reactances
        x_fd = ( 1/(self.x_td_pu - self.x_l_pu) - 1/(x_ad) )**(-1)  
        x_1q = ( 1/(self.x_tq_pu - self.x_l_pu) - 1/(x_aq) )**(-1)

        x_1d = ( 1/(self.x_std_pu - self.x_l_pu) - 1/(x_fd) - 1/(x_ad) )**(-1)
        x_2q = ( 1/(self.x_stq_pu - self.x_l_pu) - 1/(x_1q) - 1/(x_aq) )**(-1)

        # Compute rotor resistances
        r_fd = 1/(self.t_td0_s * self.w_base) * (x_ad + x_fd)
        r_1d = 1/(self.t_std0_s * self.w_base) * (x_1d + (x_ad * x_fd)/(x_ad + x_fd))

        r_1q = 1/(self.t_tq0_s * self.w_base) * (x_aq + x_1q)
        r_2q = 1/(self.t_stq0_s * self.w_base) * (x_2q + (x_aq * x_1q)/(x_aq + x_1q))

        self.x_ad_pu = x_ad
        self.x_aq_pu = x_aq
        self.x_fd_pu = x_fd
        self.x_1d_pu = x_1d
        self.x_1q_pu = x_1q
        self.x_2q_pu = x_2q
        self.r_fd_pu = r_fd
        self.r_1d_pu = r_1d
        self.r_1q_pu = r_1q
        self.r_2q_pu = r_2q


       # Define the inductance matrix
        l_d = self.x_d_pu
        l_q = self.x_q_pu
        l_0 = self.x_0_pu
        l_ad = self.x_ad_pu
        l_aq = self.x_aq_pu
        l_ffd = self.x_fd_pu + l_ad
        l_11d = self.x_1d_pu + l_ad
        l_11q = self.x_1q_pu + l_aq
        l_22q = self.x_2q_pu + l_aq

        r_a = self.r_a_pu
        r_fd = self.r_fd_pu
        r_1d = self.r_1d_pu
        r_1q = self.r_1q_pu
        r_2q = self.r_2q_pu


        L_ss = np.array( [[-l_d, 0, 0],
                          [0, -l_q, 0],
                          [0, 0, -l_0]])
        L_sr = np.array( [[l_ad, l_ad, 0, 0],
                          [0, 0, l_aq, l_aq],
                          [0, 0, 0, 0]])
        L_rs = np.array( [[-l_ad, 0, 0],
                          [-l_ad, 0, 0],
                          [0, -l_aq, 0],
                          [0, -l_aq, 0]])
        L_rr = np.array( [[l_ffd, l_ad, 0, 0],
                          [l_ad, l_11d, 0, 0],
                          [0, 0, l_11q, l_aq],
                          [0, 0, l_aq, l_22q]])

        L = np.block([[L_ss, L_sr],
                      [L_rs, L_rr]])

        Ra = np.array([[r_a, 0, 0],
                       [0, r_a, 0],
                        [0, 0, r_a]])
        Rr = -1 * np.array([[r_fd, 0, 0, 0],
                       [0, r_1d, 0, 0],
                       [0, 0, r_1q, 0],
                       [0, 0, 0, r_2q]])

        R = np.block([[Ra, np.zeros((3, 4))],
                      [np.zeros((4, 3)), Rr]])

        T = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 0]])

        T = np.block([[T, np.zeros((3, 4))],
                      [np.zeros((4, 3)), np.zeros((4, 4))]])

        self.R = R
        self.L = L
        self.T = T

    def _calculate_emt_initial_conditions(self, v_bus_mag: float, v_bus_angle: float, p_bus: float, q_bus: float) -> InitialConditionsEMT:

        # Voltage at the point of common coupling (PCC) in DQ reference frame
        v_bus_DQ = v_bus_mag * np.exp(v_bus_angle * np.pi / 180 * 1j)

        # Current sent at the PCC in DQ reference frame
        i_bus_DQ = np.conj( (p_bus + 1j * q_bus) / v_bus_DQ) 

        # Compute internal voltage generated by field circuit in the armature
        v_gen = v_bus_DQ + (self.r_a_pu + self.x_q_pu * 1j) * i_bus_DQ
        ref_angle = np.angle(v_gen) - np.pi/2 

        # Refer to the reference frame of the synchronous machine
        v_bus_dq = v_bus_DQ * np.exp(-ref_angle * 1j)
        i_bus_dq = i_bus_DQ * np.exp(-ref_angle * 1j)

        # Check: this must be zero
        v_gen_dq = v_bus_dq + (self.r_a_pu + self.x_q_pu * 1j) * i_bus_dq

        v_bus_d = np.real(v_bus_dq)
        v_bus_q = np.imag(v_bus_dq)
        i_bus_d = np.real(i_bus_dq)
        i_bus_q = np.imag(i_bus_dq)

        i_fd = (v_bus_q + self.x_d_pu * i_bus_d + self.r_a_pu * i_bus_q) / self.x_ad_pu

        v_fd = self.r_fd_pu * i_fd

        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_d, v_bus_q, 0, ref_angle)

        self.emt_init = InitialConditionsEMT(
            angle = ref_angle,
            i_d = i_bus_d,
            i_q = i_bus_q,
            i_0 = 0,
            i_fd = i_fd,
            i_1d = 0,
            i_1q = 0,
            i_2q = 0,
            v_fd = v_fd,
            v_d = v_bus_d,
            v_q = v_bus_q,
            v_0 = 0,
            v_a = v_bus_a,
            v_b = v_bus_b,
            v_c = v_bus_c
        )


    def define_variables_emt(self):
        # States 
        x = DynamicalVariables(
            name = ["i_d", "i_q", "i_0", "i_fd", "i_1d", "i_1q", "i_2q"],
            component = f"{self.type_}_{self.id}",
            init = [self.emt_init.i_d,
                    self.emt_init.i_q,
                    self.emt_init.i_0,
                    self.emt_init.i_fd,
                    self.emt_init.i_1d,
                    self.emt_init.i_1q,
                    self.emt_init.i_2q]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["v_fd", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "grid", "grid", "grid"],
            init=[  self.emt_init.v_fd,
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

    def get_derivatives_step_emt_dq0(self, i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, v_d, v_q, v_0, v_fd, w):
 
        L = self.L
        R = self.R
        T = self.T
        i = np.array([i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q])
        v = np.array([v_d, v_q, v_0, v_fd, 0, 0, 0])
        wb = self.w_base
        di_dt = np.linalg.solve(L, wb * v - wb * w * T @ L @ i + wb *R @ i)

        return di_dt

    def get_derivative_state_emt(self):

        angle, \
        i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q = self.variables_emt.x.value

        # Get inputs
        v_fd, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        # Transform currents and voltages to dq reference frame
        v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, angle)

        # Get derivatives of the state variables
        di_dt = self.get_derivatives_step_emt_dq0(i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, v_bus_d, v_bus_q, 0, v_fd, 1)

        # Angle
        dangle_dt = self.w_base

        dx_dt = np.concatenate(([dangle_dt], di_dt))

        return dx_dt

    def get_output_emt(self):
        
        angle, \
        i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q = self.variables_emt.x.value

        # Transform currents to abc reference frame
        i_bus_a, i_bus_b, i_bus_c = dq02abc(i_d, i_q, i_0, angle) 

        return [i_bus_a, i_bus_b, i_bus_c]