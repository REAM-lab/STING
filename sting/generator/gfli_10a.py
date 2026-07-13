"""
This module implements a 10th order Grid-Forming Inverter comprised of: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Current controller: A dq-based frame PI controller
- PLL: It that tracks the phase of the grid voltage.
"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field

# ------------------
# Import sting code
# ------------------
from sting.generator.core import Generator
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.transformations import dq02abc, abc2dq0
from sting.components import PhaseLockedLoop2A, InnerCurrentController2A, LCLFilter6A


@dataclass(slots=True, kw_only=True, eq=False)
class GFLI10A(Generator):
    # LCL filter parameters
    rf1_pu: float
    xf1_pu: float
    csh_pu: float
    rsh_pu: float
    txr_power_MVA: float
    txr_voltage1_kV: float
    txr_voltage2_kV: float
    txr_r1_pu: float
    txr_x1_pu: float
    txr_r2_pu: float
    txr_x2_pu: float
    # Phase-locked loop parameters
    kp_pll_pu: float
    ki_pll_puHz: float
    # Current controller parameters
    kp_cc_pu: float
    ki_cc_puHz: float
    kff_cc_pu: float

    # Components
    lcl_filter: LCLFilter6A = field(init=False)
    current_controller: InnerCurrentController2A = field(init=False)
    phase_locked_loop: PhaseLockedLoop2A = field(init=False)

    def __post_init__(self):
        self.lcl_filter = LCLFilter6A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.phase_locked_loop = PhaseLockedLoop2A(self.kp_pll_pu, self.ki_pll_puHz, self.wbase)
        self.current_controller = InnerCurrentController2A(self.kp_cc_pu, self.ki_cc_puHz, self.kff_cc_pu, self.xf1_pu + self.xf2_pu)

    @property
    def rf2_pu(self):
        return (self.txr_r1_pu + self.txr_r2_pu) * self.base_power_MVA / self.txr_power_MVA

    @property
    def xf2_pu(self):
        return (self.txr_x1_pu + self.txr_x2_pu) * self.base_power_MVA / self.txr_power_MVA
    
    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz
    

    def _calculate_emt_initial_conditions(self):
       
       

       lcl_init = self.lcl_filter.get_steady_state(
           v_bus_mag = self.power_flow_variables.vmag_bus,
           relative_phase_deg = self.power_flow_variables.vphase_bus,
           p_bus = self.power_flow_variables.p_bus,
           q_bus = self.power_flow_variables.q_bus,
       )

       self.current_controller.get_steady_state(
           v_out_d=lcl_init.v_vsc_d,
           v_out_q=lcl_init.v_vsc_q,
           v_d=lcl_init.v_bus_d,
           v_q=lcl_init.v_bus_q,
           i_d=lcl_init.i_bus_d,
           i_q=lcl_init.i_bus_q,
       )

    
    def _build_small_signal_model(self):
        # Initial conditions for the LCL filter
        init = self.lcl_filter.emt_init
        relative_phase_deg = self.power_flow_variables.vphase_bus

        # Create each components small-signal model
        cc_ssm = self.current_controller.get_small_signal_model(
            z_cc_d=self.current_controller.emt_init.z_cc_d,
            z_cc_q=self.current_controller.emt_init.z_cc_q
        )
        pll_ssm = self.phase_locked_loop.get_small_signal_model(
            v_bus_mag = self.power_flow_variables.vmag_bus,
            relative_phase_deg = relative_phase_deg
        )
        lcl_ssm = self.lcl_filter.get_small_signal_model(
            i_vsc_d=init.i_vsc_d,
            i_vsc_q=init.i_vsc_q,
            i_bus_d=init.i_bus_d,
            i_bus_q=init.i_bus_q,
            v_sh_d= init.v_sh_d,
            v_sh_q= init.v_sh_q,
        )

        # Inputs and outputs
        u = DynamicalVariables(
            name=["i_bus_d_ref", "i_bus_q_ref", "v_bus_D", "v_bus_Q"],
            type=["device", "device", "grid", "grid"],
            init=[init.i_bus_d, init.i_bus_q,init.v_bus_D, init.v_bus_Q])

        y = DynamicalVariables(
            name=['i_bus_D', 'i_bus_Q'],
            init=[init.i_bus_D, init.i_bus_Q])

        # Generate small-signal model
        components = [cc_ssm, pll_ssm, lcl_ssm]
        connections = self.get_interconnections_ssm(init.v_bus_D, init.v_bus_Q, init.i_bus_d, init.i_bus_q, relative_phase_deg)
        self.ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        return self.ssm

    def get_interconnections_ssm(self, v_bus_D, v_bus_Q, i_bus_d, i_bus_q, relative_phase_deg):

        sin = np.sin(relative_phase_deg * np.pi / 180)
        cos = np.cos(relative_phase_deg * np.pi / 180)

        R = np.array([
            [ cos,-sin],
            [ sin, cos]
        ])
        dRdt = np.array([
            [-sin,-cos],
            [ cos,-sin]
        ])

        v_D, v_Q = (dRdt.T @ np.array([[v_bus_D],[v_bus_Q]])).flatten()
        i_d, i_q = (dRdt @ np.array([[i_bus_d],[i_bus_q]])).flatten()
        
        F = np.array([
            # v_vsc_dq | delta | w | i_vsc_dq| i_bus_dq | v_f_dq 
            [0,0,  0,0,0,0,0,0,0,0], # i_ref_dq
            [0,0,  0,0,0,0,0,0,0,0], 
            [0,0,  0,0,0,0,1,0,0,0], # i_bus_dq
            [0,0,  0,0,0,0,0,1,0,0], 
            [0,0,v_D,0,0,0,0,0,0,0], # v_bus_dq
            [0,0,v_Q,0,0,0,0,0,0,0],
            [0,0,  0,0,0,0,0,0,0,0],# v_bus_DQ
            [0,0,  0,0,0,0,0,0,0,0],
            [1,0,  0,0,0,0,0,0,0,0], # v_vsc_dq
            [0,1,  0,0,0,0,0,0,0,0],
            [0,0,v_D,0,0,0,0,0,0,0], # v_bus_dq
            [0,0,v_Q,0,0,0,0,0,0,0],
            [0,0,  0,1,0,0,0,0,0,0], # w
        ])

        G = np.array([
            # i_ref_dq | v_bus_DQ
            [1,0,0,0],
            [0,1,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,R[0,0],R[1,0]],
            [0,0,R[0,1],R[1,1]],
            [0,0,1,0],
            [0,0,0,1],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,R[0,0],R[1,0]],
            [0,0,R[0,1],R[1,1]],
            [0,0,0,0]
        ])

        H = np.array([
            # v_vsc_dq | delta | w | i_vsc_dq| i_bus_dq | v_f_dq 
            [0,0,i_d,0,0,0,R[0,0],R[0,1],0,0], # i_bus_DQ
            [0,0,i_q,0,0,0,R[1,0],R[1,1],0,0],
        ])

        L = np.zeros((2,4))

        return F, G, H, L


    def define_variables_emt(self):
        # Initial conditions for the LCL filter
        init = self.lcl_filter.emt_init

        # States 
        # ------ 
        relative_phase_deg = self.power_flow_variables.vphase_bus * np.pi / 180
        z_cc_d, z_cc_q = self.current_controller.emt_init.z_cc_d, self.current_controller.emt_init.z_cc_q
        # Convert dq0 to abc 
        i_bus_a, i_bus_b, i_bus_c = dq02abc(init.i_bus_d, init.i_bus_q, 0, relative_phase_deg)
        i_vsc_a, i_vsc_b, i_vsc_c = dq02abc(init.i_vsc_d, init.i_vsc_q, 0, relative_phase_deg)
        v_sh_a, v_sh_b, v_sh_c = dq02abc(init.v_sh_d, init.v_sh_q, 0, relative_phase_deg)

        x = DynamicalVariables(
            name = ['z_cc_d', 'z_cc_q', 'theta_pll', 'gamma_pll', "i_vsc_a", "i_vsc_b", "i_vsc_c", "v_sh_a", "v_sh_b","v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c"],
            component = f"{self.type_}_{self.id}",
            init = [z_cc_d, z_cc_q, relative_phase_deg, 0, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c]
        )

        # Inputs 
        # ------
        v_bus_a, v_bus_b, v_bus_c = dq02abc(init.v_bus_D, init.v_bus_Q, 0, 0)

        u = DynamicalVariables(
            name=["i_bus_d_ref", "i_bus_q_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "grid", "grid", "grid"],
            init=[init.i_bus_d, init.i_bus_q, v_bus_a, v_bus_b, v_bus_c]
        )

        # Outputs
        # -------
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[i_bus_a, i_bus_b, i_bus_c]
        )
        
        self.variables_emt = VariablesEMT(x=x,u=u,y=y)

    def get_derivative_state_emt(self):
        """
        It returns a vector with the differential equations that describe the dynamics of the GFLI.
        This model includes: pi controller, pll, and LCL filter.
        """    
        # Get state values
        z_cc_d, z_cc_q, theta_pll, z_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value 
        
        # Get input values (external inputs)
        i_ref_d, i_ref_q, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        # convert relevant quantities to dq 
        v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, theta_pll) 
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, theta_pll) 
      
        # Update algebraic states
        v_vsc_d, v_vsc_q = self.current_controller.algebraic_step_emt_dq0(z_cc_d, z_cc_q, i_ref_d, i_ref_q, i_bus_d, i_bus_q, v_bus_d, v_bus_q)
        # Convert to abc to feed into filter dynamics 
        v_vsc_a, v_vsc_b, v_vsc_c = dq02abc(v_vsc_d, v_vsc_q, 0, theta_pll) 

        dx_cc = self.current_controller.differential_step_emt_dq0(i_ref_d, i_ref_q, i_bus_d, i_bus_q)
        dx_pll = self.phase_locked_loop.differential_step_emt_dq0(z_pll, v_bus_q)
        dx_lcl = self.lcl_filter.differential_step_emt_abc(
            i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, # states
            v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c # inputs
            )
        
        return dx_cc + dx_pll + dx_lcl
    
    def plot_results_emt(self):
        """
        Plot EMT simulation results
        """

        pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value        
        tps = self.variables_emt.x.time

        # Transform abc to dq0
        i_vsc_d, i_vsc_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, theta_pll)])
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, theta_pll)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, theta_pll)])
        
        results = DynamicalVariables(
            name=['pi_cc_d', 'pi_cc_q', 'theta_pll', 'gamma_pll', 'i_vsc_d', 'i_vsc_q', 'v_sh_d', 'v_sh_q', 'i_bus_d', 'i_bus_q'],
            component=f"{self.type_}_{self.id}",
            value=[pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q],
            time=tps
        )
        return results
    

    def get_output_emt(self):
        
        pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value

        return [i_bus_a, i_bus_b, i_bus_c]