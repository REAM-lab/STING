"""
This module implements a 25th order Grid-forming inverter model, comprised of:
- 2nd order DC circuit
- 6th order DC controller
- 1st order DC load
- 2th order inner current controller
- 2th order inner voltage controller
- 6th order LCL filter
- 2th virtual inertia
- 1th order voltage control loop
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
    InnerVoltageController2A,
    LCLFilter9A,
    RotationalInertia2A,
    VoltageDroopController1A,
    )
from sting.utils.transformations import R_dq2DQ, d_DQ2dq_dangle, d_dq2DQ_dangle

@dataclass(slots=True, kw_only=True, eq=False)
class GFMI25A(Generator):
    # LCL filter parameters
    rf1_pu: float
    xf1_pu: float
    rsh_pu: float
    csh_pu: float
    txr_power_MVA: float
    txr_voltage1_kV: float
    txr_voltage2_kV: float
    txr_r1_pu: float
    txr_x1_pu: float
    txr_r2_pu: float
    txr_x2_pu: float
    # Inner current controller parameters
    kp_cc_pu: float
    ki_cc_puHz: float
    kffv_cc: float
    # Inner voltage controller parameters
    kp_vc_pu: float
    ki_vc_puHz: float
    kffi_vc: float
    # Virtual inertia parameters
    h_s: float
    kd_pu: float
    # Voltage droop parameters
    k_q_pu: float
    w_q_puHz: float
    # DC controller parameters
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
    Ti_load_s: float # for DC/DC controller - measurement filter 
    Tload_s: float # time constant for actuation of load current change 
    i_load_ref: float 
    #Pbat_max_pu: float # maximum power capacity of battery (pu by S)
    #SOC_max_pu: float # maximum energy capacity of battery (pu by S)
    #SOC_init_pu: float # initial battery state of charge (pu)

    # Components
    lcl_filter: LCLFilter9A = field(init=False)
    voltage_controller: InnerVoltageController2A = field(init=False)
    current_controller: InnerCurrentController2A = field(init=False)
    virtual_inertia: RotationalInertia2A = field(init=False)
    voltage_droop: VoltageDroopController1A = field(init=False)
    dc_controller: DCController6A = field(init=False)
    dc_circuit: DCCircuit2A = field(init=False)
    dc_load: DCLoad1A = field(init=False)

    def __post_init__(self):
        self.lcl_filter = LCLFilter9A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.voltage_controller = InnerVoltageController2A(self.kp_vc_pu, self.ki_vc_puHz, self.kffi_vc, self.csh_pu)
        self.current_controller = InnerCurrentController2A(self.kp_cc_pu, self.ki_cc_puHz, self.kffv_cc, self.xf1_pu)
        self.virtual_inertia = RotationalInertia2A(self.h_s, self.kd_pu, self.wbase)
        self.voltage_droop = VoltageDroopController1A(self.k_q_pu, self.w_q_puHz)
        self.dc_controller = DCController6A(self.Ti_L_s, self.Tv_dc_s, self.Ti_dc_s, self.Ti_load_s, self.kp_vdc_pu, self.ki_vdc_puHz, self.kp_iL_pu, self.ki_iL_puHz, self.kff_idc, self.kff_iload)
        self.dc_circuit = DCCircuit2A(self.v_s_pu, self.l_dc_pu, self.c_dc_pu, self.wbase)
        self.dc_load = DCLoad1A(self.Tload_s)

        self.phase_angle_name = self.virtual_inertia.phase_angle_name


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
            reference_node = 'shunt'
       )

        self.voltage_controller.get_steady_state(
            i_out_d=lcl_init.i_vsc_d,
            i_out_q=lcl_init.i_vsc_q,
            i_d=lcl_init.i_bus_d,
            i_q=lcl_init.i_bus_q,
            v_d=lcl_init.v_sh_d,
            v_q=lcl_init.v_sh_q,
            w = 1
       )

        self.current_controller.get_steady_state(
           v_out_d=lcl_init.v_vsc_d,
           v_out_q=lcl_init.v_vsc_q,
           v_d=lcl_init.v_sh_d,
           v_q=lcl_init.v_sh_q,
           i_d=lcl_init.i_vsc_d,
           i_q=lcl_init.i_vsc_q,
           w = 1
       )

        self.virtual_inertia.get_steady_state(
            angle=lcl_init.angle_ref,
            w=1,
            p_ref= lcl_init.v_sh_d * lcl_init.i_bus_d + lcl_init.v_sh_q * lcl_init.i_bus_q,
        )

        self.voltage_droop.get_steady_state(
            q_ref = lcl_init.v_sh_q * lcl_init.i_bus_d - lcl_init.v_sh_d * lcl_init.i_bus_q,
            v_ref = lcl_init.v_sh_d
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

    def define_variables_emt(self):
        # States 
        x = DynamicalVariables(
            name = ["angle", "w", "q_f", "z_vc_d", "z_vc_q", "z_cc_d", "z_cc_q", "i_vsc_a", "i_vsc_b", "i_vsc_c", "v_sh_a", "v_sh_b", "v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c",
                    "i_L_f", "v_dc_f", "i_dc_f", "i_load_f","x_1", "x_2", "i_L", "v_dc", "i_load"],
            component = f"{self.type_}_{self.id}",
            init = [self.virtual_inertia.emt_init.angle,
                    self.virtual_inertia.emt_init.w,
                    self.voltage_droop.emt_init.q_ref,
                    self.voltage_controller.emt_init.z_vc_d,
                    self.voltage_controller.emt_init.z_vc_q,
                    self.current_controller.emt_init.z_cc_d,
                    self.current_controller.emt_init.z_cc_q,
                    self.lcl_filter.emt_init.i_vsc_a,
                    self.lcl_filter.emt_init.i_vsc_b,
                    self.lcl_filter.emt_init.i_vsc_c,
                    self.lcl_filter.emt_init.v_sh_a,
                    self.lcl_filter.emt_init.v_sh_b,
                    self.lcl_filter.emt_init.v_sh_c,
                    self.lcl_filter.emt_init.i_bus_a,
                    self.lcl_filter.emt_init.i_bus_b,
                    self.lcl_filter.emt_init.i_bus_c,
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
            name=["p_ref", "q_ref", "v_ref", "v_dc_ref", "v_s", "i_load_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "device", "device", "device", "device", "grid", "grid", "grid"],
            init=[  self.virtual_inertia.emt_init.p_ref,
                    self.voltage_droop.emt_init.q_ref,
                    self.voltage_droop.emt_init.v_ref,
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

        # Extract states
        angle, w, \
        q_f, \
        z_vc_d, z_vc_q, \
        z_cc_d, z_cc_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c, \
        i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, \
        i_L, v_dc, \
        i_load = x

        # Get inputs (external inputs)
        p_ref, q_ref, v_ref, v_dc_ref, v_s, i_load_ref, v_bus_a, v_bus_b, v_bus_c = u

        # Transform currents and voltages to dq reference frame
        i_vsc_d, i_vsc_q, _ = abc2dq0(i_vsc_a, i_vsc_b, i_vsc_c, angle)
        v_sh_d, v_sh_q, _ = abc2dq0(v_sh_a, v_sh_b, v_sh_c, angle)
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, angle)

        # Compute power at the shunt of the LCL filter
        p_sh = v_sh_d * i_bus_d + v_sh_q * i_bus_q
        q_sh = v_sh_q * i_bus_d - v_sh_d * i_bus_q

        # Compute voltage reference for inner voltage control loop
        u_ref_d, u_ref_q = self.voltage_droop.get_algebraics_step_emt_dq0(v_ref, q_ref, q_f)

        # Compute current reference for inner current control loop
        i_ref_d, i_ref_q = self.voltage_controller.get_algebraics_step_emt_dq0(z_vc_d, z_vc_q, u_ref_d, u_ref_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q, w)

        # Compute voltage reference for the LCL filter
        v_vsc_d, v_vsc_q = self.current_controller.get_algebraics_step_emt_dq0(z_cc_d, z_cc_q, i_ref_d, i_ref_q, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, w)

        # Transform voltage reference to abc reference frame
        v_vsc_a, v_vsc_b, v_vsc_c = dq02abc(v_vsc_d, v_vsc_q, 0, angle)

        # DC/AC power balance 
        i_dc = (v_vsc_d*i_vsc_d + v_vsc_q*i_vsc_q)/v_dc

        # Compute duty cycle for DC/DC converter
        d = self.dc_controller.get_algebraics_step_emt_dc(i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, v_dc_ref)

        # Compute derivatives of the state variables
        d_vi = self.virtual_inertia.get_derivatives_step_emt_abc(w, p_ref, p_sh)
        d_vd = self.voltage_droop.get_derivatives_step_emt_dq0(q_sh, q_f)
        d_vc = self.voltage_controller.get_derivatives_step_emt_dq0(u_ref_d, u_ref_q, v_sh_d, v_sh_q)
        d_cc = self.current_controller.get_derivatives_step_emt_dq0(i_ref_d, i_ref_q, i_vsc_d, i_vsc_q)
        d_lcl= self.lcl_filter.get_derivatives_step_emt_abc(    i_vsc_a, i_vsc_b, i_vsc_c, 
                                                                v_sh_a, v_sh_b, v_sh_c, 
                                                                i_bus_a, i_bus_b, i_bus_c, 
                                                                v_vsc_a, v_vsc_b, v_vsc_c,
                                                                v_bus_a, v_bus_b, v_bus_c)
        d_dc_controller = self.dc_controller.get_derivatives_step_emt_dc(i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, i_L, v_dc, i_dc, i_load_ref, v_dc_ref)
        d_dc_circuit = self.dc_circuit.get_derivatives_step_emt_dc(i_L, v_dc, v_s, d, i_load, i_dc)
        d_dc_load = self.dc_load.get_derivatives_step_emt_dc(i_load, i_load_ref)


        # Return derivatives of the state variables
        return d_vi + d_vd + d_vc + d_cc + d_lcl + d_dc_controller + d_dc_circuit + d_dc_load

    def get_output_emt(self, x: np.ndarray) -> np.ndarray:
        
        # Output is i_bus_abc 
        angle, w, \
        q_f, \
        z_vc_d, z_vc_q, \
        z_cc_d, z_cc_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c, \
        i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, \
        i_L, v_dc, \
        i_load = x

        return [i_bus_a, i_bus_b, i_bus_c]

    def plot_results_emt(self) -> DynamicalVariables:
        
        angle, w, \
        q_f, \
        z_vc_d, z_vc_q, \
        z_cc_d, z_cc_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c, \
        i_L_f, v_dc_f, i_dc_f, i_load_f, x_1, x_2, \
        i_L, v_dc, \
        i_load = self.variables_emt.x.value
        
        tps = self.variables_emt.x.time
        
        # Transform abc to dq0
        i_vsc_d, i_vsc_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, angle)])
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, angle)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, angle)])
        
        v_sh_dq = v_sh_d + np.multiply(v_sh_q, 1j)
        i_vsc_dq = i_vsc_d + np.multiply(i_vsc_q, 1j)
        i_bus_dq = i_bus_d + np.multiply(i_bus_q, 1j)
        v_vsc_dq = v_sh_dq + np.multiply((self.rf1_pu + self.xf1_pu * 1j), i_vsc_dq)
        p_vsc = (v_vsc_dq*np.conjugate(i_vsc_dq)).real 
        
        p_sh = (v_sh_dq*np.conjugate(i_bus_dq)).real 
        
        p_load = i_load*v_dc 
        
        results_emt = DynamicalVariables(
            name = ["angle", "w", "q_f", "z_vc_d", "z_vc_q", "z_cc_d", "z_cc_q", "i_vsc_d", "i_vsc_q", "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q", "i_L_f", 
                    "v_dc_f", "i_dc_f", "i_load_f", "x_1", "x_2", "i_L", "v_dc", "i_load", 'p_vsc', 'p_sh', 'p_load'],
            component = f"{self.type_}_{self.id}",
            value=[angle*np.pi/180, w, q_f, z_vc_d, z_vc_q, z_cc_d, z_cc_q, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q, i_L_f, 
                   v_dc_f, i_dc_f, i_load_f, x_1, x_2, i_L, v_dc, i_load, p_vsc, p_sh, p_load],
            time=tps
        )
        
        return results_emt

    def get_interconnections_ssm(self, v_bus_D, v_bus_Q, i_bus_d, i_bus_q, relative_phase_rad, v_vsc_d, v_vsc_q, i_vsc_d, i_vsc_q, v_dc, i_dc):
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
        u_stack = [u_virtual_inertia, u_voltage_droop, u_inner_voltage_controller, u_inner_current_controller, u_lcl_filter, u_dc_controller, u_dc_circuit, u_dc_load]
        y_stack = [y_virtual_inertia, y_voltage_droop, y_inner_voltage_controller, y_inner_current_controller, y_lcl_filter, y_dc_controller, y_dc_circuit, y_dc_load]
        y_sys   = [Δi_bus_D, Δi_bus_Q]
        u_sys   = [Δp_ref, Δq_ref, Δv_ref, Δv_dc_ref, Δv_s, Δi_load_ref, Δv_bus_D, Δv_bus_Q]

        note that:
        u_virtual_inertia = [Δp_ref, Δi_bus_dq, Δv_sh_dq] (5 inputs)
        u_voltage_droop = [Δq_ref, Δv_ref, Δi_bus_dq, Δv_sh_dq] (6 inputs)
        u_inner_voltage_controller = [Δv_ref_dq, Δv_sh_dq, Δi_bus_dq, Δω] (7 inputs)
        u_inner_current_controller = [Δi_ref_dq, Δi_vsc_dq, Δv_sh_dq, Δω] (7 inputs)
        u_lcl_filter = [Δv_vsc_dq, Δv_bus_dq, Δω] (5 inputs)
        u_dc_controller = [Δi_L, Δv_dc, Δi_dc, Δi_load, Δv_dc_ref] (5 inputs)
        u_dc_circuit = [Δv_s, Δd, Δi_dc, Δi_load] (4 inputs)
        u_dc_load = [Δi_load_ref] (1 input)
        
        y_virtual_inertia = [Δϕ, Δω] (2 outputs)
        y_voltage_droop = [Δv_ref_dq] (2 outputs)
        y_inner_voltage_controller = [Δi_ref_dq] (2 outputs)
        y_inner_current_controller = [Δv_ref_dq] (2 outputs)
        y_lcl_filter = [Δi_vsc_dq, Δi_bus_dq, Δv_sh_dq] (6 outputs)
        y_dc_controller = [Δd] (1 output)
        y_dc_circuit = [Δi_L, Δv_dc] (2 outputs)
        y_dc_load = [Δi_load] (1 output)

        thus: u_stack has 5 + 6 + 7 + 7 + 5 + 5 + 4 + 1 = 40 inputs, y_stack has 2 + 2 + 2 + 2 + 6 + 1 + 2 + 1= 18 outputs, y_sys has 2 outputs, and u_sys has 8 inputs.

        
        Interconnection matrices
        ------------------------
        Recall that to linearize the transformation from DQ to dq (and vice versa)
            Δv_dq = Uᵀ*(v_DQ)ₒ*Δϕ + Rᵀ*Δv_DQ 
            Δi_DQ = U *(i_dq)ₒ*Δϕ + R *Δi_dq 
        where
            R = [ cosϕₒ  -sinϕₒ ]
                [ sinϕₒ   cosϕₒ ]
            U = d/dϕₒ R 
        and we will define
            a := Uᵀ*(v_DQ)ₒ
            b := U *(i_dq)ₒ

        ┌ component ──▶                 |   APC    ┆ RPC        ┆ IVC        ┆ ICC        ┆ LCL                            ┆ DC controller  ┆ DC circuit      ┆ DC load     │ Device inputs        Grid inputs
        │       ┌ index ──▶             │   0   1  ┆ 2,3        ┆ 4,5        ┆ 6,7        ┆ 8,9        10,11      12,13    ┆ 14             ┆ 15        16    ┆ 17          │ 0         1       2       3           4       5               6,7 
        ▼       ▼                       │   Δϕ  Δω ┆ Δv_ref_dq  ┆ Δi_ref_dq  ┆ Δv_vsc_dq  ┆ Δi_vsc_dq  Δi_bus_dq  Δv_sh_dq ┆ Δd             ┆ Δi_L      Δv_dc ┆ Δi_load     │ Δp_ref    Δq_ref  Δv_ref  Δv_dc_ref   Δv_s    Δi_load_ref     Δv_bus_DQ
        ────────────────────────────────┼──────────┴────────────┴────────────┴────────────┴────────────────────────────────┴────────────────┴─────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────
        APC         0       Δp_ref      │   0   0    0            0            0            0          0          0          0                0         0       0           │   1        0        0        0        0       0               0     
                    1,2     Δi_bus_dq   │   0   0    0            0            0            0          I₂         0          0                0         0       0           │   0        0        0        0        0       0               0     
                    3,4     Δv_sh_dq    │   0   0    0            0            0            0          0          I₂         0                0         0       0           │   0        0        0        0        0       0               0     
        RPC         5       Δq_ref      │   0   0    0            0            0            0          0          0          0                0         0       0           │   0        1        0        0        0       0               0           
                    6       Δv_ref      │   0   0    0            0            0            0          0          0          0                0         0       0           │   0        0        1        0        0       0               0     
                    7,8     Δi_bus_dq   │   0   0    0            0            0            0          I₂         0          0                0         0       0           │   0        0        0        0        0       0               0
                    9,10    Δv_sh_dq    │   0   0    0            0            0            0          0          I₂         0                0         0       0           │   0        0        0        0        0       0               0
        IVC         11,12   Δv_ref_dq   │   0   0    I₂           0            0            0          0          0          0                0         0       0           │   0        0        0        0        0       0               0     
                    13,14   Δv_sh_dq    │   0   0    0            0            0            0          0          I₂         0                0         0       0           │   0        0        0        0        0       0               0
                    15,16   Δi_bus_dq   │   0   0    0            0            0            0          I₂         0          0                0         0       0           │   0        0        0        0        0       0               0 
                    17      Δω          │   0   1    0            0            0            0          0          0          0                0         0       0           │   0        0        0        0        0       0               0
        ICC         18,19   Δi_ref_dq   │   0   0    0            I₂           0            0          0          0          0                0         0       0           │   0        0        0        0        0       0               0                         
                    20,21   Δi_vsc_dq   │   0   0    0            0            0            I₂         0          0          0                0         0       0           │   0        0        0        0        0       0               0 
                    22,23   Δv_sh_dq    │   0   0    0            0            0            0          0          I₂         0                0         0       0           │   0        0        0        0        0       0               0
                    24      Δω          │   0   1    0            0            0            0          0          0          0                0         0       0           │   0        0        0        0        0       0               0        
        LCL         25,26   Δv_vsc_dq   │   0   0    0            0            I₂           0          0          0          0                0         0       0           │   0        0        0        0        0       0               0     
                    27,28   Δv_bus_dq   │   a   0    0            0            0            0          0          0          0                0         0       0           │   0        0        0        0        0       0               Rᵀ     
                    29      Δω          │   0   1    0            0            0            0          0          0          0                0         0       0           │   0        0        0        0        0       0               0   
        DC contr.   30      Δi_L        │   0   0    0            0            0            0          0          0          0                1         0       0           │   0        0        0        0        0       0               0
                    31      Δv_dc       │   0   0    0            0            0            0          0          0          0                0         1       0           │   0        0        0        0        0       0               0
                    32      Δi_dc       │   0   0    0            0            [a2, a4]     [a1, a3]   0          0          0                0         a5      0           │   0        0        0        0        0       0               0
                    33      Δi_load     │   0   0    0            0            0            0          0          0          0                0         0       1           │   0        0        0        0        0       0               0
                    34      Δv_dc_ref   │   0   0    0            0            0            0          0          0          0                0         0       0           │   0        0        0        1        0       0               0
        DC circ.    35      Δv_s        │   0   0    0            0            0            0          0          0          0                0         0       0           │   0        0        0        0        1       0               0
                    36      Δd          │   0   0    0            0            0            0          0          0          1                0         0       0           │   0        0        0        0        0       0               0
                    37      Δi_dc       │   0   0    0            0            [a2, a4]     [a1, a3]   0          0          0                0         a5      0           │   0        0        0        0        0       0               0
                    38      Δi_load     │   0   0    0            0            0            0          0          0          0                0         0       1           │   0        0        0        0        0       0               0
        DC load     39      Δi_load_ref │   0   0    0            0            0            0          0          0          0                0         0       0           │   0        0        0        0        0       1               0    
        ────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────
        Grid        0,1     Δi_bus_DQ   │   b   0    0            0            0            0          R          0          0                0         0       0           │   0        0        0        0        0       0               0
        outputs
        """
        angle = relative_phase_rad
        R = R_dq2DQ(angle)
        I = np.eye(2)

        a = d_DQ2dq_dangle(v_bus_D, v_bus_Q, angle).reshape(2,1)
        b = d_dq2DQ_dangle(i_bus_d, i_bus_q, angle).reshape(2,1)

        F = np.zeros((40, 18))
        G = np.zeros((40, 8))
        H = np.zeros((2, 18))
        L = np.zeros((2, 8))


        a1 = v_vsc_d/v_dc
        a2 = i_vsc_d/v_dc
        a3 = v_vsc_q/v_dc  
        a4 = i_vsc_q/v_dc 
        a5 = -i_dc/v_dc 

        idx_F =[
            ([1,2], [10,11], I), ([3,4], [12,13], I), ([7,8], [10,11], I), ([9,10], [12,13], I),
            ([11,12], [2,3], I), ([13,14], [12,13], I), ([15,16], [10,11], I), ([17], [1], 1),
            ([18,19], [4,5], I), ([20,21], [8,9], I), ([22,23], [12,13], I),  ([24], [1], 1), 
            ([25,26], [6,7], I), ([27,28], [0], a), ([29], [1], 1), 
            ([30], [15], 1), ([31], [16], 1), ([32], [6], a2), ([32], [7], a4), ([32], [8], a1), ([32], [9], a3), ([32], [16], a5),
            ([33], [17], 1),
            ([36], [14], 1),
            ([37], [6], a2), ([37], [7], a4), ([37], [8], a1), ([37], [9], a3), ([37], [16], a5),
            ([38], [17], 1),
        ]
        for rows, cols, value in idx_F:
            F[np.ix_(rows, cols)] = value

        idx_G = [
            ([0], [0], 1), ([5,6],[1,2], I), ([27,28], [6,7], R.T), 
            ([34], [3], 1), ([35], [4], 1),
            ([39], [5], 1)
        ]
        for rows, cols, value in idx_G:
            G[np.ix_(rows, cols)] = value
        
        H[:, [0]] = b
        H[np.ix_([0,1], [10,11])] = R

        return F, G, H, L

    def _build_small_signal_model(self):


        # Create each components small-signal model
        virtual_inertia_ssm = self.virtual_inertia.get_small_signal_model(
            i_d = self.lcl_filter.emt_init.i_bus_d,
            i_q = self.lcl_filter.emt_init.i_bus_q,
            v_d = self.lcl_filter.emt_init.v_sh_d,
            v_q = self.lcl_filter.emt_init.v_sh_q,
            angle = self.virtual_inertia.emt_init.angle,
            p_ref = self.virtual_inertia.emt_init.p_ref
        )

        voltage_droop_ssm = self.voltage_droop.get_small_signal_model(
            i_d = self.lcl_filter.emt_init.i_bus_d,
            i_q = self.lcl_filter.emt_init.i_bus_q,
            v_d = self.lcl_filter.emt_init.v_sh_d,
            v_q = self.lcl_filter.emt_init.v_sh_q,
            q_ref = self.voltage_droop.emt_init.q_ref,
            v_ref = self.voltage_droop.emt_init.v_ref
        )

        voltage_controller_ssm = self.voltage_controller.get_small_signal_model(
            z_vc_d = self.voltage_controller.emt_init.z_vc_d,
            z_vc_q = self.voltage_controller.emt_init.z_vc_q,
            v_d = self.lcl_filter.emt_init.v_sh_d,
            v_q = self.lcl_filter.emt_init.v_sh_q,
            i_d = self.lcl_filter.emt_init.i_bus_d,
            i_q = self.lcl_filter.emt_init.i_bus_q,
            w = 1
        )

        current_controller_ssm = self.current_controller.get_small_signal_model(
            z_cc_d = self.current_controller.emt_init.z_cc_d,
            z_cc_q = self.current_controller.emt_init.z_cc_q,
            i_d = self.lcl_filter.emt_init.i_vsc_d,
            i_q = self.lcl_filter.emt_init.i_vsc_q,
            v_d = self.lcl_filter.emt_init.v_sh_d,
            v_q = self.lcl_filter.emt_init.v_sh_q,
            w = 1
        )

        lcl_filter_ssm = self.lcl_filter.get_small_signal_model(
            i_vsc_d = self.lcl_filter.emt_init.i_vsc_d,
            i_vsc_q = self.lcl_filter.emt_init.i_vsc_q,
            i_bus_d = self.lcl_filter.emt_init.i_bus_d,
            i_bus_q = self.lcl_filter.emt_init.i_bus_q,
            v_sh_d = self.lcl_filter.emt_init.v_sh_d,
            v_sh_q = self.lcl_filter.emt_init.v_sh_q
        )

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

        # Inputs and outputs
        u = DynamicalVariables(
            name=["p_ref", "q_ref", "v_ref", "v_dc_ref", "v_s", "i_load_ref", "v_bus_D", "v_bus_Q"],
            type=["device", "device", "device", "device", "device", "device", "grid", "grid"],
            init=[self.virtual_inertia.emt_init.p_ref,
                  self.voltage_droop.emt_init.q_ref,
                  self.voltage_droop.emt_init.v_ref,
                  self.v_dc_ref,
                  self.v_s_pu,
                  self.i_load_ref,
                  self.lcl_filter.emt_init.v_bus_D,
                  self.lcl_filter.emt_init.v_bus_Q])

        y = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            init=[self.lcl_filter.emt_init.i_bus_D, 
                  self.lcl_filter.emt_init.i_bus_Q])

        # Generate small-signal model
        components = [virtual_inertia_ssm, voltage_droop_ssm, voltage_controller_ssm, current_controller_ssm, lcl_filter_ssm, dc_controller_ssm, dc_circuit_ssm, dc_load_ssm]
        connections = self.get_interconnections_ssm(self.lcl_filter.emt_init.v_bus_D, 
                                                    self.lcl_filter.emt_init.v_bus_Q,
                                                    self.lcl_filter.emt_init.i_bus_d, 
                                                    self.lcl_filter.emt_init.i_bus_q,
                                                    self.lcl_filter.emt_init.angle_ref,
                                                     self.lcl_filter.emt_init.v_vsc_d,
                                                     self.lcl_filter.emt_init.v_vsc_q,
                                                     self.lcl_filter.emt_init.i_vsc_d,
                                                     self.lcl_filter.emt_init.i_vsc_q,
                                                     self.dc_circuit.emt_init.v_dc,
                                                     self.dc_circuit.emt_init.i_dc)
        self.ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        return self.ssm