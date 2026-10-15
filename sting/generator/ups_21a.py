"""

The model is based on the paper ``Dynamic Modeling of Data-Center
Power Delivery for Power System Resonance Analysis``.  
Includes AFE, DC link, VSI, PSU array, and downstream DC-DC/load 
"""

from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import numpy as np
import sympy as sp
from sympy.matrices.dense import matrix2numpy

from sting.generator.core import Generator
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel
from sting.utils.transformations import abc2dq0, dq02abc

class InitialConditionsEMT(NamedTuple):
    v_eq: float
    gload: float 
    i_eq: float 
    xi_eq: float 
    v_psu: float 
    i_psu: float 
    v_vsi_u: float 
    v_vsi_v: float 
    g_eq: float 
    i_vsi_u: float 
    i_vsi_v: float 
    w_vsi: float 
    i_cv_u: float 
    i_cv_v: float 
    xi_vsi_u: float 
    xi_vsi_v: float 
    gamma_vsi_u: float 
    gamma_vsi_v: float 
    m_u: float 
    m_v: float 
    m_d: float 
    m_q: float 
    i_afe_d: float 
    i_afe_q: float 
    v_dc_ups: float
    gamma_afe_d: float 
    gamma_afe_q: float 
    xi_dc_afe: float 
    vq_pll: float 
    w_pll: float 
    epsilon_pll: float 
    v_pcc_d: float 
    v_pcc_q: float 
    theta_pll: float
    xi_psu: float
    pload: float 
    v_bus_D: float 
    v_bus_Q: float
    

@dataclass(slots=True, kw_only=True, eq=False)
class UPS21A(Generator):
    """Online double-conversion UPS and data-center load equivalent."""
    
    # Base operating-point references.
    v_dc_ups_ref: float = 1.0
    v_vsi_u_ref: float = 1.0
    v_psu_ref: float = 1.0
    v_eq_ref: float = 0.5
    w_s: float = 1.0

    # AFE and PLL.
    l_afe: float = 0.05
    r_afe: float = 0.003
    w_pll_lp: float = 2 * np.pi * 100.0
    k_pll_p: float = 177.7
    k_pll_i: float = 15791.367041742973
    k_dc_afe_p: float = 0.333
    k_dc_afe_i: float = 5.236
    k_c_afe_p: float = 0.233
    k_c_afe_i: float = 209.4

    # Shared DC link.
    c_dc: float = 2.0

    # VSI output filter and controllers.
    l_vsi: float = 0.05
    r_vsi: float = 0.003
    c_vsi: float = 0.2
    k_v_vsi_p: float = 0.667
    k_v_vsi_i: float = 209.4
    k_c_vsi_p: float = 0.664
    k_c_vsi_i: float = 837.8

    # PSU array common-DC equivalent.
    c_psu: float = 2.0
    r_psu: float = 0.005
    k_v_psu_p: float = 0.667
    k_v_psu_i: float = 20.94

    # Downstream DC-DC/load equivalent.
    c_eq: float = 0.2
    k_v_eq_p: float = 0.667
    k_v_eq_i: float = 209.4
    
    emt_init: Optional[InitialConditionsEMT] = None

    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz

    @staticmethod
    def _state_names():
        return [
            "theta_pll", "epsilon_pll", "vq_pll", "i_afe_d", "i_afe_q",
            "xi_dc_afe", "gamma_afe_d", "gamma_afe_q", "v_dc_ups", "i_cv_u", "i_cv_v",
            "v_vsi_u", "v_vsi_v", "xi_vsi_u", "xi_vsi_v", "gamma_vsi_u",
            "gamma_vsi_v", "v_psu", "xi_psu", "v_eq", "xi_eq",
        ]
        
    def _set_control_gains(self):
        
        def compute_gains(fbw, zeta, C_or_L, type, R=0.0):
            omega_n = 2.0 * np.pi * fbw
            if type == "V":
                kp = 2.0 * zeta * omega_n * C_or_L / self.wbase
                ki = (omega_n ** 2) * C_or_L / self.wbase
            elif type == "C":
                kp = 2.0 * zeta * omega_n * C_or_L / self.wbase -R 
                ki = (omega_n ** 2) * C_or_L / self.wbase
            return kp, ki 
        
        def compute_pll_gains(fbw, zeta):
            omega_n = 2.0 * np.pi * fbw
            kp = 2.0 * zeta * omega_n
            ki = omega_n ** 2
            return kp, ki
        
        self.k_dc_afe_p, self.k_dc_afe_i = compute_gains(fbw=5, zeta=1.0, C_or_L=self.c_dc, type="V")
        self.k_c_afe_p, self.k_c_afe_i = compute_gains(fbw=200, zeta=0.707, C_or_L=self.l_afe, type="C", R=self.r_afe)
        self.k_v_vsi_p, self.k_v_vsi_i = compute_gains(fbw=80, zeta=1.0, C_or_L=self.c_vsi, type="V")
        self.k_c_vsi_p, self.k_c_vsi_i = compute_gains(fbw=400, zeta=1.0, C_or_L=self.l_vsi, type="C", R=self.r_vsi) 
        self.k_v_psu_p, self.k_v_psu_i = compute_gains(fbw=10, zeta=1, C_or_L=self.c_psu, type="V")
        self.k_v_eq_p, self.k_v_eq_i = compute_gains(fbw=100, zeta=1, C_or_L=self.c_eq, type="V")
        self.k_pll_p, self.k_pll_i = compute_pll_gains(fbw=20, zeta=0.707)


    def _calculate_emt_initial_conditions(self):

        self._set_control_gains()
        """Initializes states from power flow solution and prints state tables."""
        # Given values
        w_vsi = w_pll = self.w_s
        v_dc_ups = self.v_dc_ups_ref
        v_vsi_u = self.v_vsi_u_ref
        v_vsi_v = 0.0
        v_psu = self.v_psu_ref
        v_eq = self.v_eq_ref
        v_vsi_norm_2 = v_vsi_u**2 + v_vsi_v**2
        vq_pll = 0.0
        epsilon_pll = 0.0

        # Read powerflow solution 
        vmag_bus = self.power_flow_variables.vmag_bus
        vphase_bus = self.power_flow_variables.vphase_bus
        p_bus = self.power_flow_variables.p_bus
        q_bus = self.power_flow_variables.q_bus

        v_bus_DQ = vmag_bus * np.exp(1j * np.radians(vphase_bus))
        theta_pll = np.angle(v_bus_DQ)

        v_pcc_dq = v_bus_DQ * np.exp(-1j * theta_pll)
        v_pcc_d, v_pcc_q = v_pcc_dq.real, v_pcc_dq.imag

        # convert to consumption = positive convention 
        p_afe = -p_bus 
        q_afe = -q_bus
        i_bus_DQ = (p_afe - 1j * q_afe) / np.conj(v_bus_DQ)
        
        i_afe_dq = i_bus_DQ * np.exp(-1j * theta_pll)
        i_afe_d = i_afe_dq.real
        i_afe_q = i_afe_dq.imag
        xi_dc_afe = i_afe_d / self.k_dc_afe_i

        # compute power loss in AFE 
        p_afe_loss = self.r_afe * (i_afe_d**2 + i_afe_q**2)
        i_dc_in = (p_afe - p_afe_loss) / v_dc_ups
        assert i_dc_in > 0, f"i_dc_in={i_dc_in}: UPS should be consuming; check p_bus sign"

        # Use KVL to compute v_afe_dq
        v_afe_d = v_pcc_d - self.r_afe * i_afe_d + w_pll * self.l_afe * i_afe_q
        v_afe_q = v_pcc_q - self.r_afe * i_afe_q - w_pll * self.l_afe * i_afe_d

        gamma_afe_d = (v_pcc_d - self.r_afe * i_afe_d) / self.k_c_afe_i
        gamma_afe_q = (v_pcc_q - self.r_afe * i_afe_q) / self.k_c_afe_i
        m_d, m_q = v_afe_d / v_dc_ups, v_afe_q / v_dc_ups

        # 4. VSI Bridge Computations (Quadratic Solution for i_cv_u)
        i_cv_v = w_vsi * self.c_vsi * v_vsi_u
        coeff_a = self.r_vsi
        coeff_b = v_vsi_u
        coeff_c = self.r_vsi * (i_cv_v**2) - v_dc_ups * i_dc_in

        roots = np.roots([coeff_a, coeff_b, coeff_c])
        i_cv_u = np.max(roots)  # TODO ? 

        m_u = (v_vsi_u + self.r_vsi * i_cv_u - w_vsi * self.l_vsi * i_cv_v) / v_dc_ups
        m_v = (v_vsi_v + self.r_vsi * i_cv_v + w_vsi * self.l_vsi * i_cv_u) / v_dc_ups
        i_dc_out = m_u * i_cv_u + m_v * i_cv_v

        gamma_vsi_u = (v_vsi_u + self.r_vsi * i_cv_u) / self.k_c_vsi_i
        gamma_vsi_v = (v_vsi_v + self.r_vsi * i_cv_v) / self.k_c_vsi_i
        i_vsi_u, i_vsi_v = i_cv_u, 0.0

        # PSU and Load 
        g_eq = i_vsi_u / v_vsi_u
        xi_vsi_u = i_cv_u / self.k_v_vsi_i
        xi_vsi_v = 0.0

        i_psu = ((g_eq - self.r_psu * (g_eq**2))* v_vsi_norm_2) / (3.0 * v_psu)
        xi_psu = g_eq / self.k_v_psu_i

        i_eq = i_psu * v_psu / v_eq
        gload = i_eq / v_eq
        xi_eq = i_eq / self.k_v_eq_i
        pload = gload * 3.0 * (v_eq**2)

        self.emt_init = InitialConditionsEMT(
            v_eq=v_eq,
            gload=gload,
            i_eq=i_eq,
            xi_eq=xi_eq,
            v_psu=v_psu,
            i_psu=i_psu,
            v_vsi_v=v_vsi_v,
            v_vsi_u=v_vsi_u,
            g_eq=g_eq,
            i_vsi_u=i_vsi_u,
            i_vsi_v=i_vsi_v,
            w_vsi=w_vsi,
            i_cv_u=i_cv_u,
            i_cv_v=i_cv_v,
            xi_vsi_u=xi_vsi_u,
            xi_vsi_v=xi_vsi_v,
            gamma_vsi_u=gamma_vsi_u,
            gamma_vsi_v=gamma_vsi_v,
            m_u=m_u,
            m_v=m_v,
            m_d=m_d,
            m_q=m_q,
            i_afe_d=i_afe_d,
            i_afe_q=i_afe_q,
            v_dc_ups=v_dc_ups,
            gamma_afe_d=gamma_afe_d,
            gamma_afe_q=gamma_afe_q,
            xi_dc_afe=xi_dc_afe,
            vq_pll=vq_pll,
            w_pll=w_pll,
            epsilon_pll=epsilon_pll,
            v_pcc_d=v_pcc_d,
            v_pcc_q=v_pcc_q,
            theta_pll=theta_pll,
            xi_psu=xi_psu,
            pload=pload,
            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag
        )


    def plot_results_emt(self) -> DynamicalVariables:
      
        tps = self.variables_emt.x.time
                
        theta_pll, epsilon_pll, vq_pll, i_afe_d, i_afe_q, xi_dc_afe, gamma_afe_d, gamma_afe_q, v_dc_ups, i_cv_u, i_cv_v, v_vsi_u, v_vsi_v, xi_vsi_u, xi_vsi_v, gamma_vsi_u, gamma_vsi_v,  v_psu, xi_psu, v_eq, xi_eq = self.variables_emt.x.value
        
        # compute additional algebraic states
        w_pll = self.w_s + self.k_pll_p * vq_pll + self.k_pll_i * epsilon_pll
        v_vsi_norm_2 = np.square(v_vsi_u) + np.square(v_vsi_v)
        i_vsi_u = i_cv_u
        g_eq = i_vsi_u / v_vsi_u
        i_psu = ((g_eq - self.r_psu * (g_eq**2))* v_vsi_norm_2) / (3.0 * v_psu)
        i_eq  = i_psu * v_psu / v_eq
        
        # input values aren't updated in variables_emt.u so can't do this yet 
        # v_dc_ups_ref, v_vsi_u_ref, v_psu_ref, pload, v_eq_ref, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u 
        # v_bus_D, v_bus_Q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_bus_a, v_bus_b, v_bus_c, theta_pll)])
        # # convert to converter frame 
        # c_ang = np.cos(theta_pll)
        # s_ang = np.sin(theta_pll)
                
        # v_pcc_d =  c_ang * v_bus_D + s_ang * v_bus_Q
        # v_pcc_q = -s_ang * v_bus_D + c_ang * v_bus_Q
        # p_pcc = v_pcc_d * i_afe_d + v_pcc_q * i_afe_q
        # q_pcc = v_pcc_q * i_afe_d - v_pcc_d * i_afe_q
                
        results_emt = DynamicalVariables(
            name = ["theta_pll", "epsilon_pll", "vq_pll", "i_afe_d", "i_afe_q", "xi_dc_afe", "gamma_afe_d", "gamma_afe_q", "v_dc_ups", "i_cv_u", "i_cv_v", "v_vsi_u", "v_vsi_v", "xi_vsi_u", "xi_vsi_v", "gamma_vsi_u", "gamma_vsi_v",  "v_psu", "xi_psu", "v_eq", "xi_eq", "w_pll", "i_psu", "i_eq"],
            component = f"{self.type_}_{self.id}",
            value=[theta_pll, epsilon_pll, vq_pll, i_afe_d, i_afe_q, xi_dc_afe, gamma_afe_d, gamma_afe_q, v_dc_ups, i_cv_u, i_cv_v, v_vsi_u, v_vsi_v, xi_vsi_u, xi_vsi_v, gamma_vsi_u, gamma_vsi_v,  v_psu, xi_psu, v_eq, xi_eq, w_pll, i_psu, i_eq],
            time=tps
        )
                
        return results_emt

    def _build_small_signal_model(self):
        """Build symbolic A/B/C/D matrices from the active EMT equations."""
        x = sp.symbols(" ".join(self._state_names()))
        u = sp.symbols("v_dc_ups_ref v_vsi_u_ref v_psu_ref pload v_eq_ref v_bus_D v_bus_Q")
        
        (theta_pll, epsilon_pll, vq_pll, i_afe_d, i_afe_q, xi_dc_afe, gamma_afe_d, gamma_afe_q, v_dc_ups,
         i_cv_u, i_cv_v, v_vsi_u, v_vsi_v, xi_vsi_u, xi_vsi_v, gamma_vsi_u, gamma_vsi_v,
          v_psu, xi_psu, v_eq, xi_eq) = x
        
        v_dc_ups_ref, v_vsi_u_ref, v_psu_ref, pload, v_eq_ref, v_bus_D, v_bus_Q = u
        
        v_vsi_v_ref = 0
        
        wb = self.wbase
        c_ang = sp.cos(theta_pll)
        s_ang = sp.sin(theta_pll)
        
        # grid to converter transform 
        # [cos, sin; -sin, cos]
        v_pcc_d =  c_ang * v_bus_D + s_ang * v_bus_Q
        v_pcc_q = -s_ang * v_bus_D + c_ang * v_bus_Q
        
        
        w_pll = self.w_s + self.k_pll_p * vq_pll + self.k_pll_i * epsilon_pll
        
        i_afe_d_ref = (self.k_dc_afe_p * (v_dc_ups_ref - v_dc_ups) + self.k_dc_afe_i * xi_dc_afe)
        i_afe_q_ref = 0 
        
        v_afe_d_ref = self.k_c_afe_p * (i_afe_d - i_afe_d_ref) + self.k_c_afe_i * gamma_afe_d + w_pll * self.l_afe * i_afe_q 
        v_afe_q_ref = self.k_c_afe_p * i_afe_q + self.k_c_afe_i * gamma_afe_q - w_pll * self.l_afe * i_afe_d
        
        m_afe_d = v_afe_d_ref / v_dc_ups
        m_afe_q = v_afe_q_ref / v_dc_ups
        
        w_vsi = self.w_s
        
        i_cv_u_ref = self.k_v_vsi_p * (v_vsi_u_ref - v_vsi_u) + self.k_v_vsi_i * xi_vsi_u - w_vsi * self.c_vsi * v_vsi_v
        i_cv_v_ref = self.k_v_vsi_p * (-v_vsi_v) + self.k_v_vsi_i * xi_vsi_v + w_vsi * self.c_vsi * v_vsi_u
        
        v_cv_u_ref = self.k_c_vsi_p * (i_cv_u_ref - i_cv_u) + self.k_c_vsi_i * gamma_vsi_u - w_vsi * self.l_vsi * i_cv_v
        v_cv_v_ref = self.k_c_vsi_p * (i_cv_v_ref - i_cv_v) + self.k_c_vsi_i * gamma_vsi_v + w_vsi * self.l_vsi * i_cv_u
        
        m_vsi_u = v_cv_u_ref / v_dc_ups 
        m_vsi_v = v_cv_v_ref / v_dc_ups
        
        g_eq = self.k_v_psu_p * (v_psu_ref - v_psu) + self.k_v_psu_i * xi_psu
        
        i_vsi_u = g_eq * v_vsi_u
        i_vsi_v = g_eq * v_vsi_v
        
        g_load = pload / (3 * v_eq_ref**2)
        
        i_eq = self.k_v_eq_p * (v_eq_ref - v_eq) + self.k_v_eq_i * xi_eq
        
        i_psu = v_eq / v_psu * i_eq
        
        i_dc_in = m_afe_d * i_afe_d + m_afe_q * i_afe_q
        i_dc_out = m_vsi_u * i_cv_u + m_vsi_v * i_cv_v
        psu_injection_term = ((g_eq - self.r_psu * (g_eq**2)) * (v_vsi_u**2 + v_vsi_v**2)) / (3 * v_psu)
            
        
        f = sp.Matrix([
            (w_pll - self.w_s) * wb, # theta_pll  
            vq_pll, # epsilon_pll 
            self.w_pll_lp * (v_pcc_q - vq_pll), # vq_pll  
            (wb / self.l_afe) * (v_pcc_d - v_dc_ups * m_afe_d - self.r_afe * i_afe_d + w_pll * self.l_afe * i_afe_q), # i_afe_d 
            (wb / self.l_afe) * (v_pcc_q - v_dc_ups * m_afe_q - self.r_afe * i_afe_q - w_pll * self.l_afe * i_afe_d), # i_afe_q 
            v_dc_ups_ref - v_dc_ups, # dv_dc_ups 
            i_afe_d - i_afe_d_ref, # gamma_afe_d 
            i_afe_q - i_afe_q_ref, # gamma_afe_q 
            (wb / self.c_dc) * (i_dc_in - i_dc_out), # v_dc_ups 
            (wb / self.l_vsi) * (v_dc_ups * m_vsi_u - v_vsi_u - self.r_vsi * i_cv_u + w_vsi * self.l_vsi * i_cv_v), # i_cv_u 
            (wb / self.l_vsi) * (v_dc_ups * m_vsi_v - v_vsi_v - self.r_vsi * i_cv_v - w_vsi * self.l_vsi * i_cv_u), #i_cv_v 
            (wb / self.c_vsi) * (i_cv_u - i_vsi_u + w_vsi * self.c_vsi * v_vsi_v), #v_vsi_u 
            (wb / self.c_vsi) * (i_cv_v - i_vsi_v - w_vsi * self.c_vsi * v_vsi_u), #v_vsi_v 
            v_vsi_u_ref - v_vsi_u, #xi_vsi_u 
            v_vsi_v_ref - v_vsi_v, #xi_vsi_v 
            i_cv_u_ref - i_cv_u, #gamma_vsi_u
            i_cv_v_ref - i_cv_v, #gamma_vsi_v
            (wb / self.c_psu) * (psu_injection_term - i_psu), #v_psu 
            v_psu_ref - v_psu, #xi_psu
            (wb / self.c_eq) * (i_eq - g_load * v_eq), #v_eq
            v_eq_ref - v_eq #xi_eq
        ])
        
        # Outputs 
        
        # transform from converter to grid is:
        # [cos, -sin; sin, cos]

        i_afe_D = c_ang * i_afe_d - s_ang * i_afe_q
        i_afe_Q = s_ang * i_afe_d + c_ang * i_afe_q
        
        # minus signs because these values are computed using consumption-positive convention, but rest of the system uses injection-positive convention 
        y = sp.Matrix([-i_afe_D, -i_afe_Q])
        
        # Inputs 
        matrix_inputs = [v_dc_ups_ref, v_vsi_u_ref, v_psu_ref, pload, v_eq_ref, v_bus_D, v_bus_Q]
        
        x0 = np.array([
            self.emt_init.theta_pll, self.emt_init.epsilon_pll, self.emt_init.vq_pll,
            self.emt_init.i_afe_d, self.emt_init.i_afe_q, self.emt_init.xi_dc_afe,
            self.emt_init.gamma_afe_d, self.emt_init.gamma_afe_q, self.emt_init.v_dc_ups, self.emt_init.i_cv_u,
            self.emt_init.i_cv_v, self.emt_init.v_vsi_u, self.emt_init.v_vsi_v,
            self.emt_init.xi_vsi_u, self.emt_init.xi_vsi_v, self.emt_init.gamma_vsi_u,
            self.emt_init.gamma_vsi_v, self.emt_init.v_psu,
            self.emt_init.xi_psu, self.emt_init.v_eq, self.emt_init.xi_eq
        ]) 
        
        u0 = np.array([self.v_dc_ups_ref, self.v_vsi_u_ref, self.v_psu_ref, self.emt_init.pload, self.v_eq_ref, self.emt_init.v_bus_D, self.emt_init.v_bus_Q])
        
        # simplify the construction of the numeric A, B, C, D matrices
        # construct the numeric A, B, C, D matrices by evaluating the Jacobians at the operating point
        
        A_1 = f.jacobian(x)
        A_2 = A_1.subs(dict(zip(x, x0)))
        A = A_2.subs(dict(zip(u, u0)))
        
        B_1 = f.jacobian(matrix_inputs)
        B_2 = B_1.subs(dict(zip(x, x0)))
        B = B_2.subs(dict(zip(u, u0)))
        
        C_1 = y.jacobian(x)
        C_2 = C_1.subs(dict(zip(x, x0)))
        C = C_2.subs(dict(zip(u, u0)))

        D_1 = y.jacobian(matrix_inputs)
        D_2 = D_1.subs(dict(zip(x, x0)))
        D = D_2.subs(dict(zip(u, u0)))
        
        A = matrix2numpy(A, dtype=float)
        B = matrix2numpy(B, dtype=float)
        C = matrix2numpy(C, dtype=float)
        D = matrix2numpy(D, dtype=float)
        
        component = f"{self.type_}_{self.id}"
        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D,
            x=DynamicalVariables(name=self._state_names(), component=component, init=x0),
            u=DynamicalVariables(name=["v_dc_ups_ref", "v_vsi_u_ref", "v_psu_ref", "pload", "v_eq_ref", "v_bus_D", "v_bus_Q"], component=component, type=["device"] * 5 + ["grid", "grid"], init=u0),
            y=DynamicalVariables(name=["i_bus_D", "i_bus_Q"], component=component))
        return self.ssm

    def define_variables_emt(self):
        
        x = DynamicalVariables(
            name = ["theta_pll", "epsilon_pll", "vq_pll", "i_afe_d", "i_afe_q", "xi_dc_afe", "gamma_afe_d", "gamma_afe_q", "v_dc_ups", "i_cv_u", "i_cv_v", "v_vsi_u", "v_vsi_v", "xi_vsi_u", "xi_vsi_v", "gamma_vsi_u", "gamma_vsi_v", "v_psu", "xi_psu", "v_eq", "xi_eq"],
            component = f"{self.type_}_{self.id}",
            init = [self.emt_init.theta_pll, 
                    self.emt_init.epsilon_pll,
                    self.emt_init.vq_pll,
                    self.emt_init.i_afe_d,
                    self.emt_init.i_afe_q,
                    self.emt_init.xi_dc_afe,
                    self.emt_init.gamma_afe_d,
                    self.emt_init.gamma_afe_q,
                    self.emt_init.v_dc_ups,
                    self.emt_init.i_cv_u,
                    self.emt_init.i_cv_v,
                    self.emt_init.v_vsi_u,
                    self.emt_init.v_vsi_v,
                    self.emt_init.xi_vsi_u,
                    self.emt_init.xi_vsi_v,
                    self.emt_init.gamma_vsi_u,
                    self.emt_init.gamma_vsi_v,
                    self.emt_init.v_psu,
                    self.emt_init.xi_psu,
                    self.emt_init.v_eq,
                    self.emt_init.xi_eq]
        )

        # Inputs 
        
        v_bus_a, v_bus_b, v_bus_c = dq02abc(self.emt_init.v_bus_D, self.emt_init.v_bus_Q, 0.0, self.emt_init.theta_pll)
        
        u = DynamicalVariables(
            name=["v_dc_ups_ref", "v_vsi_u_ref", "v_psu_ref", "pload", "v_eq_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "device", "device", "device", "grid", "grid", "grid"],
            init=[self.v_dc_ups_ref, 
                  self.v_vsi_u_ref,
                  self.v_psu_ref, 
                  self.emt_init.pload,
                  self.v_eq_ref,
                  v_bus_a, 
                  v_bus_b, 
                  v_bus_c]
        )

        # Outputs
        
        # minus signs because of the internal convention (consumption positive) v system convention of injection positive
        i_bus_a, i_bus_b, i_bus_c = dq02abc(-self.emt_init.i_afe_d, -self.emt_init.i_afe_q, 0.0, self.emt_init.theta_pll)
        
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[i_bus_a, 
                  i_bus_b, 
                  i_bus_c]
        )
        
        self.variables_emt = VariablesEMT(x=x,u=u,y=y)

    def get_derivative_state_emt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Get states
        
        theta_pll, epsilon_pll, vq_pll, i_afe_d, i_afe_q, xi_dc_afe, gamma_afe_d, gamma_afe_q, v_dc_ups, i_cv_u, i_cv_v, v_vsi_u, v_vsi_v, xi_vsi_u, xi_vsi_v, gamma_vsi_u, gamma_vsi_v, v_psu, xi_psu, v_eq, xi_eq = x 
        
        # Get input values
        v_dc_ups_ref, v_vsi_u_ref, v_psu_ref, pload, v_eq_ref, v_bus_a, v_bus_b, v_bus_c = u 
        
        # Convert to dq frame of AFE 
        v_pcc_d, v_pcc_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, theta_pll) 
        
        ## AFE 
        
        # Algebraic equations 
        w_pll = self.w_s + self.k_pll_p*vq_pll + self.k_pll_i*epsilon_pll
        i_afe_d_ref = self.k_dc_afe_p*(v_dc_ups_ref - v_dc_ups) + self.k_dc_afe_i*xi_dc_afe
        i_afe_q_ref = 0
        
        v_afe_d_ref = self.k_c_afe_p*(i_afe_d - i_afe_d_ref) + self.k_c_afe_i*gamma_afe_d + w_pll*self.l_afe*i_afe_q 
        v_afe_q_ref = self.k_c_afe_p*(i_afe_q - i_afe_q_ref) + self.k_c_afe_i*gamma_afe_q - w_pll*self.l_afe*i_afe_d 
        
        m_d = v_afe_d_ref/v_dc_ups
        m_q = v_afe_q_ref/v_dc_ups
        
        # Dynamic equations 
        d_theta_pll = (w_pll - self.w_s)*self.wbase #todo is this correct for emt?
        d_theta_pll = w_pll*self.wbase
        d_epsilon_pll = vq_pll
        d_vq_pll = (v_pcc_q - vq_pll)*self.w_pll_lp
        
        d_i_afe_d = (v_pcc_d - v_dc_ups*m_d - self.r_afe*i_afe_d + w_pll*self.l_afe*i_afe_q)*self.wbase/self.l_afe
        d_i_afe_q = (v_pcc_q - v_dc_ups*m_q - self.r_afe*i_afe_q - w_pll*self.l_afe*i_afe_d)*self.wbase/self.l_afe
        
        d_xi_dc_afe = v_dc_ups_ref - v_dc_ups 
        
        d_gamma_afe_d = i_afe_d - i_afe_d_ref
        d_gamma_afe_q = i_afe_q - i_afe_q_ref 
        
        ## VSI 
        
        w_vsi = self.w_s 
        v_vsi_v_ref = 0 
        
        i_cv_u_ref = (self.k_v_vsi_p*(v_vsi_u_ref - v_vsi_u) + self.k_v_vsi_i*xi_vsi_u - w_vsi*self.c_vsi*v_vsi_v) #check sign
        i_cv_v_ref = (self.k_v_vsi_p*(v_vsi_v_ref - v_vsi_v) + self.k_v_vsi_i*xi_vsi_v + w_vsi*self.c_vsi*v_vsi_u)
        
        v_cv_u_ref = (self.k_c_vsi_p*(i_cv_u_ref - i_cv_u) + self.k_c_vsi_i*gamma_vsi_u - w_vsi*self.l_vsi*i_cv_v) #check sign
        v_cv_v_ref = (self.k_c_vsi_p*(i_cv_v_ref - i_cv_v) + self.k_c_vsi_i*gamma_vsi_v + w_vsi*self.l_vsi*i_cv_u)
        
        m_u = v_cv_u_ref/v_dc_ups
        m_v = v_cv_v_ref/v_dc_ups         
        
        d_i_cv_u = (v_dc_ups*m_u - v_vsi_u - self.r_vsi*i_cv_u + w_vsi*self.l_vsi*i_cv_v)*self.wbase/self.l_vsi #check sign
        d_i_cv_v = (v_dc_ups*m_v - v_vsi_v - self.r_vsi*i_cv_v - w_vsi*self.l_vsi*i_cv_u)*self.wbase/self.l_vsi

        g_eq = self.k_v_psu_p*(v_psu_ref - v_psu) + self.k_v_psu_i*xi_psu 
        i_vsi_u = g_eq*v_vsi_u 
        i_vsi_v = g_eq*v_vsi_v 
        
        d_v_vsi_u = (i_cv_u - i_vsi_u + w_vsi*self.c_vsi*v_vsi_v)*self.wbase/self.c_vsi #check sign 
        d_v_vsi_v = (i_cv_v - i_vsi_v - w_vsi*self.c_vsi*v_vsi_u)*self.wbase/self.c_vsi
        
        d_xi_vsi_u = v_vsi_u_ref - v_vsi_u 
        d_xi_vsi_v = v_vsi_v_ref - v_vsi_v  
        
        d_gamma_vsi_u = i_cv_u_ref - i_cv_u 
        d_gamma_vsi_v = i_cv_v_ref - i_cv_v 

        gload = pload/(3*v_eq_ref**2)
        i_eq = self.k_v_eq_p*(v_eq_ref - v_eq) + self.k_v_eq_i*xi_eq
        i_psu = (v_eq/v_psu)*i_eq
        
        ## Capacitor
        i_dc_in = m_d*i_afe_d + m_q*i_afe_q 
        i_dc_out = m_u*i_cv_u + m_v*i_cv_v 
        d_v_dc_ups = (i_dc_in - i_dc_out)*self.wbase/self.c_dc
        
        ## PSU 
        v_vsi_norm_2 = v_vsi_u**2 + v_vsi_v**2
        d_v_psu = (((g_eq - self.r_psu*g_eq**2)*(v_vsi_norm_2))/(3*v_psu) - i_psu)*self.wbase/self.c_psu 
        d_xi_psu = v_psu_ref - v_psu 

        ## DC-DC aggregated
        d_v_eq = (i_eq - gload*v_eq)*(self.wbase/self.c_eq )
        d_xi_eq = v_eq_ref - v_eq 

        return np.array([d_theta_pll, d_epsilon_pll, d_vq_pll, d_i_afe_d, d_i_afe_q, d_xi_dc_afe, d_gamma_afe_d, d_gamma_afe_q, d_v_dc_ups, d_i_cv_u, d_i_cv_v, d_v_vsi_u, d_v_vsi_v, d_xi_vsi_u, d_xi_vsi_v, d_gamma_vsi_u, d_gamma_vsi_v, d_v_psu, d_xi_psu, d_v_eq, d_xi_eq ])
        
        

    def get_output_emt(self, x: np.ndarray) -> np.ndarray:
        
        theta_pll, epsilon_pll, vq_pll, i_afe_d, i_afe_q, xi_dc_afe, gamma_afe_d, gamma_afe_q, v_dc_ups, i_cv_u, i_cv_v, v_vsi_u, v_vsi_v, xi_vsi_u, xi_vsi_v, gamma_vsi_u, gamma_vsi_v,  v_psu, xi_psu, v_eq, xi_eq = x 
        
        # negative sign because of sign convention 
        i_afe_a, i_afe_b, i_afe_c = dq02abc(-i_afe_d, -i_afe_q, 0.0, theta_pll)
        
        return [i_afe_a, i_afe_b, i_afe_c]
