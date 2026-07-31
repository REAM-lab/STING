"""
This module implements a 10th order Grid-following Inverter comprised of: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Current controller: A dq-based frame PI controller
- PLL: It that tracks the phase of the grid voltage.


NOTE: SSM is in progress, CCM matrices need to be updated
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
class GFLI13A(Generator):
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
    kff_cc: float

    # Components
    lcl_filter: LCLFilter6A = field(init=False)
    current_controller: InnerCurrentController2A = field(init=False)
    phase_locked_loop: PhaseLockedLoop2A = field(init=False)

    def __post_init__(self):
        self.lcl_filter = LCLFilter6A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.phase_locked_loop = PhaseLockedLoop2A(self.kp_pll_pu, self.ki_pll_puHz, self.wbase)
        self.current_controller = InnerCurrentController2A(self.kp_cc_pu, self.ki_cc_puHz, self.kff_cc, self.xf1_pu + self.xf2_pu)

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
           reference_node = 'bus'
       )

        self.current_controller.get_steady_state(
           v_out_d=lcl_init.v_vsc_d,
           v_out_q=lcl_init.v_vsc_q,
           v_d=lcl_init.v_bus_d,
           v_q=lcl_init.v_bus_q,
           i_d=lcl_init.i_bus_d,
           i_q=lcl_init.i_bus_q,
           w = 1
       )

        self.phase_locked_loop.get_steady_state(
            v_mag = self.power_flow_variables.vmag_bus,
            relative_phase_deg = self.power_flow_variables.vphase_bus
        )

    
    def _build_small_signal_model(self):
        # Unpack OPF solutions
        v_mag, phase_deg = self.power_flow_variables.vmag_bus, self.power_flow_variables.vphase_bus
        # Initial conditions in the LCL filter
        i_bus_d, i_bus_q = self.lcl_filter.emt_init.i_bus_d, self.lcl_filter.emt_init.i_bus_q
        i_vsc_d, i_vsc_q = self.lcl_filter.emt_init.i_vsc_d, self.lcl_filter.emt_init.i_vsc_q
        v_sh_d, v_sh_q = self.lcl_filter.emt_init.v_sh_d, self.lcl_filter.emt_init.v_sh_q
        v_bus_d, v_bus_q = self.lcl_filter.emt_init.v_bus_d, self.lcl_filter.emt_init.v_bus_q

        z_cc_d, z_cc_q = self.current_controller.emt_init.z_cc_d, self.current_controller.emt_init.z_cc_q

        # Create each components small-signal model
        pll_ssm = self.phase_locked_loop.get_small_signal_model(
            v_bus_mag=v_mag, relative_phase_deg=phase_deg)
        cc_ssm = self.current_controller.get_small_signal_model(
            z_cc_d=z_cc_d, z_cc_q=z_cc_q, i_d=i_bus_d, i_q=i_bus_q, v_d=v_bus_d, v_q=v_bus_q, w=1)
        lcl_ssm = self.lcl_filter.get_small_signal_model(
            i_vsc_d=i_vsc_d, i_vsc_q=i_vsc_q, i_bus_d=i_bus_d, i_bus_q=i_bus_q, v_sh_d=v_sh_d, v_sh_q=v_sh_q)

        # Inverter level inputs and outputs
        v_bus_D, v_bus_Q = self.lcl_filter.emt_init.v_bus_D, self.lcl_filter.emt_init.v_bus_Q
        i_bus_D, i_bus_Q = self.lcl_filter.emt_init.i_bus_D, self.lcl_filter.emt_init.i_bus_Q
        # Inputs and outputs
        u = DynamicalVariables(
            name=["i_bus_d_ref", "i_bus_q_ref", "v_bus_D", "v_bus_Q"],
            type=["device", "device", "grid", "grid"],
            init=[i_bus_d, i_bus_q, v_bus_D, v_bus_Q])

        y = DynamicalVariables(
            name=['i_bus_D', 'i_bus_Q'],
            init=[i_bus_D, i_bus_Q])

        # Generate small-signal model
        components = [cc_ssm, pll_ssm, lcl_ssm]
        connections = self.get_interconnections_ssm(v_bus_D, v_bus_Q, i_bus_d, i_bus_q, phase_deg)
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

        # States 
        x = DynamicalVariables(
            name = ['z_cc_d', 'z_cc_q', 'theta_pll', 'gamma_pll', "i_vsc_a", "i_vsc_b", "i_vsc_c", "v_sh_a", "v_sh_b","v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c"],
            component = f"{self.type_}_{self.id}",
            init = [self.current_controller.emt_init.z_cc_d, 
                    self.current_controller.emt_init.z_cc_q,
                    self.phase_locked_loop.emt_init.theta_pll, 
                    self.phase_locked_loop.emt_init.z_pll,
                    self.lcl_filter.emt_init.i_vsc_a, self.lcl_filter.emt_init.i_vsc_b, self.lcl_filter.emt_init.i_vsc_c,
                    self.lcl_filter.emt_init.v_sh_a, self.lcl_filter.emt_init.v_sh_b, self.lcl_filter.emt_init.v_sh_c,
                    self.lcl_filter.emt_init.i_bus_a, self.lcl_filter.emt_init.i_bus_b, self.lcl_filter.emt_init.i_bus_c]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["i_ref_d", "i_ref_q", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "grid", "grid", "grid"],
            init=[ self.lcl_filter.emt_init.i_bus_d,
                  self.lcl_filter.emt_init.i_bus_q,
                  self.lcl_filter.emt_init.v_bus_a, 
                  self.lcl_filter.emt_init.v_bus_b, 
                  self.lcl_filter.emt_init.v_bus_c]
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[self.lcl_filter.emt_init.i_bus_a, 
                  self.lcl_filter.emt_init.i_bus_b, 
                  self.lcl_filter.emt_init.i_bus_c]
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

        # convert relevant quantities to dq (reference frame of the IBR)
        v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, theta_pll) 
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, theta_pll) 

        # Compute time derivatives of PLL
        d_x_pll = self.phase_locked_loop.get_derivatives_step_emt_abc(theta_pll, z_pll, # states in PLL
                                                                      v_bus_a, v_bus_b, v_bus_c # inputs to PLL
                                                                      )

        # Compute frequency estimated by PLL
        w_pll  = d_x_pll[0]/self.wbase
      
        # Compute the voltage references from the inner current controller. No delay assumed in VSC.
        v_vsc_d, v_vsc_q = self.current_controller.get_algebraics_step_emt_dq0(z_cc_d, z_cc_q, # states in current controller
                                                                               i_ref_d, i_ref_q, i_bus_d, i_bus_q, v_bus_d, v_bus_q, w_pll # inputs to current controller
                                                                               )

        # Compute the time derivatives of the current controller
        d_x_cc = self.current_controller.get_derivatives_step_emt_dq0(i_ref_d, i_ref_q, i_bus_d, i_bus_q) # inputs to current controller

        # Convert to abc to feed into filter dynamics 
        v_vsc_a, v_vsc_b, v_vsc_c = dq02abc(v_vsc_d, v_vsc_q, 0, theta_pll) 

        # Compute the time derivatives of the LCL filter
        d_x_lcl = self.lcl_filter.get_derivatives_step_emt_abc(
            i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, # states in LCL filter
            v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c # inputs to LCL filter
            )
        
        return d_x_cc + d_x_pll + d_x_lcl
    
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