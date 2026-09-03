"""
This module implements a GFLI that incorporates: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Current controller: A dq-based frame PI controller
- PLL: It that tracks the phase of the grid voltage. The PLL has a proportional and integral gain.
- DC-DC converter for controlling Vdc
- DC side circuit with load modeled as current source 
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
from sting.components import (
    DCCircuit2A,
    DCController6A,
    DCLoad1A,
    InnerCurrentController2A,
    LCLFilter9A,
    PhaseLockedLoop2A
    )
from sting.utils.transformations import R_dq2DQ, d_DQ2dq_dangle, d_dq2DQ_dangle

@dataclass(slots=True, kw_only=True, eq=False)
class GFLI23A(Generator):
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
    # PLL 
    kp_pll_rad_s: float
    ki_pll_rad2_s2: float
    # CC 
    kp_cc_pu: float
    ki_cc_puHz: float
    kff_cc: float
    # DC controller and circuit and load 
    kp_vdc_pu: float # DC voltage regulator P gain 
    ki_vdc_puHz: float # DC voltage regulator I gain [1/s] 
    kp_iL_pu: float # DC current regulator P gain 
    ki_iL_puHz: float # DC current regulator I gain [1/s]
    l_dc_pu: float # DC/DC converter inductance, [pu]
    c_dc_pu: float # DC link capacitance, [pu]
    v_dc_ref: float # DC bus reference voltage, [pu]
    v_s_pu: float # DC voltage source voltage, [pu]
    Ti_L_s: float # measurement filter time constants [s]
    Tv_dc_s: float 
    Ti_dc_s: float 
    kff_idc: float 
    kff_iload: float 
    i_load_ref: float 
    Ti_load_s: float # for DC/DC controller - measurement filter 
    Tload_s: float # time constant for xactuation of load current change 
    
    # Components
    lcl_filter: LCLFilter9A = field(init=False)
    current_controller: InnerCurrentController2A = field(init=False)
    phase_locked_loop: PhaseLockedLoop2A = field(init=False)
    dc_controller: DCController6A = field(init=False)
    dc_circuit: DCCircuit2A = field(init=False)
    dc_load: DCLoad1A = field(init=False)

    @property
    def rf2_pu(self):
        return (self.txr_r1_pu + self.txr_r2_pu) * self.base_power_MVA / self.txr_power_MVA

    @property
    def xf2_pu(self):
        return (self.txr_x1_pu + self.txr_x2_pu) * self.base_power_MVA / self.txr_power_MVA
    
    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz
    
    def __post_init__(self):
            self.lcl_filter = LCLFilter9A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
            self.phase_locked_loop = PhaseLockedLoop2A(self.kp_pll_rad_s, self.ki_pll_rad2_s2, self.wbase)
            self.current_controller = InnerCurrentController2A(self.kp_cc_pu, self.ki_cc_puHz, self.kff_cc, self.xf1_pu + self.xf2_pu)
            self.dc_controller = DCController6A(self.Ti_L_s, self.Tv_dc_s, self.Ti_dc_s, self.Ti_load_s, self.kp_vdc_pu, self.ki_vdc_puHz, self.kp_iL_pu, self.ki_iL_puHz, self.kff_idc, self.kff_iload)
            self.dc_circuit = DCCircuit2A(self.v_s_pu, self.l_dc_pu, self.c_dc_pu, self.wbase)
            self.dc_load = DCLoad1A(self.Tload_s) 
                
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
            
            self.dc_circuit.get_steady_state(
                        p = lcl_init.v_vsc_d * lcl_init.i_vsc_d + lcl_init.v_vsc_q * lcl_init.i_vsc_q,
                        v_dc = self.v_dc_ref,
                        i_load = self.i_load_ref
                    )
            
            self.dc_controller.get_steady_state(
                i_L = self.dc_circuit.emt_init.i_L,
                i_dc = self.dc_circuit.emt_init.i_dc,
                i_load = self.i_load_ref,
                d = self.dc_circuit.emt_init.d,
                v_dc = self.v_dc_ref
            )
    
            self.dc_load.get_steady_state(
                i_load = self.i_load_ref
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
        
        dc_controller_ssm = self.dc_controller.get_small_signal_model(
                    i_dc = self.dc_circuit.emt_init.i_dc,
                    i_L = self.dc_circuit.emt_init.i_L,
                    v_dc = self.dc_circuit.emt_init.v_dc,
                    i_load = self.i_load_ref,
                    x_1 = self.dc_controller.emt_init.x_1,
                    x_2 = self.dc_controller.emt_init.x_2,
                    d = self.dc_circuit.emt_init.d
                )
        
        dc_circuit_ssm = self.dc_circuit.get_small_signal_model(
            d = self.dc_circuit.emt_init.d,
            i_L = self.dc_circuit.emt_init.i_L,
            v_dc = self.dc_circuit.emt_init.v_dc)

        dc_load_ssm = self.dc_load.get_small_signal_model(
            i_load = self.dc_load.emt_init.i_load)

        # Inverter level inputs and outputs
        v_bus_D, v_bus_Q = self.lcl_filter.emt_init.v_bus_D, self.lcl_filter.emt_init.v_bus_Q
        i_bus_D, i_bus_Q = self.lcl_filter.emt_init.i_bus_D, self.lcl_filter.emt_init.i_bus_Q
        # Inputs and outputs
        u = DynamicalVariables(
            name=["i_ref_d", "i_ref_q", "v_dc_ref", "v_s", "i_load_ref", "v_bus_D", "v_bus_Q"],
            type=["device", "device", "device","device","device", "grid", "grid"],
            init=[i_bus_d, i_bus_q, self.v_dc_ref, self.v_s_pu, self.i_load_ref, v_bus_D, v_bus_Q])

        y = DynamicalVariables(
            name=['i_bus_D', 'i_bus_Q'],
            init=[i_bus_D, i_bus_Q])

        # Generate small-signal model
        components = [pll_ssm, cc_ssm, lcl_ssm, dc_controller_ssm, dc_circuit_ssm, dc_load_ssm]
        connections = self.get_interconnections_ssm(v_bus_D, v_bus_Q, i_bus_d, i_bus_q, phase_deg,self.lcl_filter.emt_init.v_vsc_d,
                                                             self.lcl_filter.emt_init.v_vsc_q,
                                                             self.lcl_filter.emt_init.i_vsc_d,
                                                             self.lcl_filter.emt_init.i_vsc_q,
                                                             self.dc_circuit.emt_init.v_dc,
                                                             self.dc_circuit.emt_init.i_dc)
        self.ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        return self.ssm
    
    def get_interconnections_ssm(self, v_bus_D, v_bus_Q, i_bus_d, i_bus_q, relative_phase_deg, v_vsc_d, v_vsc_q, i_vsc_d, i_vsc_q, v_dc, i_dc):
        """
        Construct the interconnection matrices F, H, G, and L that satisfies:
        u_stack = F * y_stack + H * u_sys
        y_sys   = G * y_stack + L * u_sys

        Given the tableau form:

                │   y_stack  │   u_sys
        ───────────────────────────────────────────────
        u_stack │   F        │   G
        ───────────────────────────────────────────────
        y_sys   │   H        │   L
        
        where:
            u_stack = [u_pll, u_inner_current_controller, u_lcl_filter, u_dc_controller, u_dc_circuit, u_dc_load]
            y_stack = [y_pll, y_inner_current_controller, y_lcl_filter, y_dc_controller, y_dc_circuit,y_dc_load]
            y_sys   = [Δi_bus_D, Δi_bus_Q]
            u_sys   = [Δi_d_ref, Δi_q_ref, Δv_dc_ref, Δv_s, Δi_load_ref, Δv_bus_D, Δv_bus_Q]
    
            note that:
            u_pll = [Δv_bus_D, Δv_bus_Q] (2 inputs)
            u_inner_current_controller = [Δi_ref_dq, Δi_vsc_dq, Δv_sh_dq, Δω] (7 inputs)
            u_lcl_filter = [Δv_vsc_dq, Δv_bus_dq, Δω] (5 inputs)
            u_dc_controller = [Δi_L, Δv_dc, Δi_dc, Δi_load, Δv_dc_ref] (5 inputs)
            u_dc_circuit = [Δv_s, Δd, Δi_dc, Δi_load] (4 inputs)
            u_dc_load = [Δi_load_ref] (1 input)
            
            y_pll = [Δϕ, Δω] (2 outputs)
            y_inner_current_controller = [Δv_ref_dq] (2 outputs)
            y_lcl_filter = [Δi_vsc_dq, Δi_bus_dq, Δv_sh_dq] (6 outputs)
            y_dc_controller = [Δd] (1 output)
            y_dc_circuit = [Δi_L, Δv_dc] (2 outputs)
            y_dc_load = [Δi_load] (1 output)
    
            thus: 
            u_stack has 2 + 7 + 5 + 5 + 4 + 1 = 24 inputs 
            y_stack has 2 + 2 + 6 + 1 + 2 + 1 = 14 outputs
            y_sys has 2 outputs
            u_sys has 7 inputs
        
        """

        angle = relative_phase_deg * np.pi / 180
        R = R_dq2DQ(angle)
        I = np.eye(2)

        a = d_DQ2dq_dangle(v_bus_D, v_bus_Q, angle).reshape(2,1)
        b = d_dq2DQ_dangle(i_bus_d, i_bus_q, angle).reshape(2,1)

        F = np.zeros((24, 14))
        G = np.zeros((24, 7))
        H = np.zeros((2, 14))
        L = np.zeros((2, 7))
        
        a1 = v_vsc_d/v_dc
        a2 = i_vsc_d/v_dc
        a3 = v_vsc_q/v_dc  
        a4 = i_vsc_q/v_dc 
        a5 = -i_dc/v_dc 

        # Entries in F and G entered as tuples: (row_idx, col_idx, values)
        idx_F =[
            ([4,5], [6, 7], I), ([6,7], [0], a), ([8], [1], 1), ([9,10], [2,3], I), ([11,12], [1], a), ([13], [1], 1),
            ([14],[11], 1), ([15], [12], 1), ([16], [2],a2), ([16],[3], a4), ([16], [4], a1), ([16], [5], a3), ([16], [12], a5),
            ([21], [2],a2), ([21],[3], a4), ([21], [4], a1), ([21], [5], a3), ([21], [12], a5),
            ([17],[13], 1), ([20], [10], 1), ([22],[13], 1)
            ]
        for rows, cols, value in idx_F:
            F[np.ix_(rows, cols)] = value
        
        idx_G = [
            ([2], [0], 1), ([3], [1], 1), ([0,1], [5,6], I), 
            ([6,7], [5,6], R.T), ([11,12], [5,6], R.T),
            ([18],[2], 1), ([19], [3], 1), ([23], [4], 1)
            ]
        for rows, cols, value in idx_G:
            G[np.ix_(rows, cols)] = value
        # Add values to H
        H[:,[0]] = b
        H[np.ix_([0,1],[6,7])] = R

        return (F,G,H,L)
        
    def define_variables_emt(self):

        # States 
        x = DynamicalVariables(
            name = ["z_cc_d", "z_cc_q", "theta_pll", "gamma_pll", "i_vsc_a", "i_vsc_b", "i_vsc_c", "v_sh_a", "v_sh_b","v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c", "i_L_f", "v_dc_f", "i_dc_f", "i_load_f","x_1", "x_2", "i_L", "v_dc", "i_load"],
            component = f"{self.type_}_{self.id}",
            init = [self.current_controller.emt_init.z_cc_d, 
                    self.current_controller.emt_init.z_cc_q,
                    self.phase_locked_loop.emt_init.theta_pll, 
                    self.phase_locked_loop.emt_init.z_pll,
                    self.lcl_filter.emt_init.i_vsc_a, self.lcl_filter.emt_init.i_vsc_b, self.lcl_filter.emt_init.i_vsc_c,
                    self.lcl_filter.emt_init.v_sh_a, self.lcl_filter.emt_init.v_sh_b, self.lcl_filter.emt_init.v_sh_c,
                    self.lcl_filter.emt_init.i_bus_a, self.lcl_filter.emt_init.i_bus_b, self.lcl_filter.emt_init.i_bus_c,
                    self.dc_controller.emt_init.i_L_f,
                                        self.dc_controller.emt_init.v_dc_f,
                                        self.dc_controller.emt_init.i_dc_f,
                                        self.dc_controller.emt_init.i_load_f,
                                        self.dc_controller.emt_init.x_1,
                                        self.dc_controller.emt_init.x_2,
                                        self.dc_circuit.emt_init.i_L,
                                        self.dc_circuit.emt_init.v_dc,
                                        self.dc_load.emt_init.i_load]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["i_ref_d", "i_ref_q", "v_dc_ref", "v_s", "i_load_ref","v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "device", "device", "device", "grid", "grid", "grid"],
            init=[ self.lcl_filter.emt_init.i_bus_d,
                  self.lcl_filter.emt_init.i_bus_q,
                  self.v_dc_ref,
                  self.v_s_pu,
                  self.i_load_ref,
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
        
    def get_derivative_state_emt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        
        # Get state values
        z_cc_d, z_cc_q, theta_pll, z_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, i_L, v_dc, i_load = x
        
        # Get input values (external inputs)
        i_ref_d, i_ref_q, v_dc_ref, v_s, i_load_ref, v_bus_a, v_bus_b, v_bus_c = u

        # convert relevant quantities to dq (reference frame of the IBR)
        v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, theta_pll) 
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, theta_pll)
        i_vsc_d, i_vsc_q, _ = abc2dq0(i_vsc_a, i_vsc_b, i_vsc_c, theta_pll)

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
        
        # DC/AC power balance 
        i_dc = (v_vsc_d*i_vsc_d + v_vsc_q*i_vsc_q)/v_dc

        # Compute duty cycle for DC/DC converter
        d = self.dc_controller.get_algebraics_step_emt_dc(i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, v_dc_ref)

        # Compute the time derivatives of the LCL filter
        d_x_lcl = self.lcl_filter.get_derivatives_step_emt_abc(
            i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, # states in LCL filter
            v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c # inputs to LCL filter
            )
        d_dc_controller = self.dc_controller.get_derivatives_step_emt_dc(i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, i_L, v_dc, i_dc, i_load_ref, v_dc_ref)
        d_dc_circuit = self.dc_circuit.get_derivatives_step_emt_dc(i_L, v_dc, v_s, d, i_load, i_dc)
        d_dc_load = self.dc_load.get_derivatives_step_emt_dc(i_load, i_load_ref)
        
        return d_x_cc + d_x_pll + d_x_lcl + d_dc_controller + d_dc_circuit + d_dc_load
    
    def get_output_emt(self, x: np.ndarray) -> np.ndarray:

        # Output is i_bus_abc 
        z_cc_d, z_cc_q, theta_pll, z_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, i_L, v_dc, i_load = x
        
        return [i_bus_a, i_bus_b, i_bus_c]
    
    def plot_results_emt(self):
        
        """
        Plot EMT simulation results
        """

        z_cc_d, z_cc_q, theta_pll, z_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, i_L, v_dc, i_load = self.variables_emt.x.value        
        tps = self.variables_emt.x.time

        # Transform abc to dq0
        i_vsc_d, i_vsc_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, theta_pll)])
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, theta_pll)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, theta_pll)])
        
        results = DynamicalVariables(
            name=["z_cc_d", "z_cc_q", "theta_pll", "gamma_pll", "i_vsc_d", "i_vsc_q", "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q", "i_L_f", "v_dc_f", "i_dc_f", "i_load_f","x_1", "x_2", "i_L", "v_dc", "i_load"],
            component=f"{self.type_}_{self.id}",
            value=[z_cc_d, z_cc_q, theta_pll, z_pll, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q, i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, i_L, v_dc, i_load],
            time=tps
        )
        return results
        
