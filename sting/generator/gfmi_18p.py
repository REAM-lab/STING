"""
This module implements a 18th order Grid-Forming Inverter comprised of: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Current controller: A dq-based frame PI controller
- Voltage controller: A dq-based frame PI controller
- Virtual inertia (active power control): A virtual inertia control that emulates the behavior of a synchronous generator.
- Voltage droop (reactive power control): A droop control that emulates the behavior of a synchronous generator.
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
from sting.utils.transformations import dq02abc, abc2dq0, R_dq2DQ, R_DQ2dq, d_dq2DQ_dangle, d_DQ2dq_dangle
from sting.components import LCLFilter6A, InnerVoltageController2A, InnerCurrentController2A, VirtualInertia2A, VoltageDroopController1A

@dataclass(slots=True, kw_only=True, eq=False)
class GFMI18P(Generator):
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
    # Output-feedback control
    F: np.ndarray

    # Components
    lcl_filter: LCLFilter6A = field(init=False)
    voltage_controller: InnerVoltageController2A = field(init=False)
    current_controller: InnerCurrentController2A = field(init=False)
    virtual_inertia: VirtualInertia2A = field(init=False)
    voltage_droop: VoltageDroopController1A = field(init=False)


    def __post_init__(self):
        self.lcl_filter = LCLFilter6A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.voltage_controller = InnerVoltageController2A(self.kp_vc_pu, self.ki_vc_puHz, self.kffi_vc, self.csh_pu)
        self.current_controller = InnerCurrentController2A(self.kp_cc_pu, self.ki_cc_puHz, self.kffv_cc, self.xf1_pu)
        self.virtual_inertia = VirtualInertia2A(self.h_s, self.kd_pu, self.wbase)
        self.voltage_droop = VoltageDroopController1A(self.k_q_pu, self.w_q_puHz)

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
        
    def define_variables_emt(self):
        # States 
        x = DynamicalVariables(
            name = ["angle", "w", "q_f", "z_vc_d", "z_vc_q", "z_cc_d", "z_cc_q", "i_vsc_a", "i_vsc_b", "i_vsc_c", "v_sh_a", "v_sh_b", "v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c"],
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
                    self.lcl_filter.emt_init.i_bus_c]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["p_ref", "q_ref", "v_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "device", "grid", "grid", "grid"],
            init=[  self.virtual_inertia.emt_init.p_ref,
                    self.voltage_droop.emt_init.q_ref,
                    self.voltage_droop.emt_init.v_ref,
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

    def output_feedback_control_emt(self, w, i_vsc_d, i_vsc_q, i_bus_d, i_bus_q):

        delta_y = np.array([w - 1, 
                            i_vsc_d - self.lcl_filter.emt_init.i_vsc_d, 
                            i_vsc_q - self.lcl_filter.emt_init.i_vsc_q, 
                            i_bus_d - self.lcl_filter.emt_init.i_bus_d, 
                            i_bus_q - self.lcl_filter.emt_init.i_bus_q])
        delta_u = self.F @ delta_y

        delta_p_ref, delta_q_ref, delta_v_ref = delta_u

        return delta_p_ref, delta_q_ref, delta_v_ref

    def get_derivative_state_emt(self):

        # Extract states
        angle, w, \
        q_f, \
        z_vc_d, z_vc_q, \
        z_cc_d, z_cc_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value

        # Get inputs
        p_ref, q_ref, v_ref, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        # Transform currents and voltages to dq reference frame
        i_vsc_d, i_vsc_q, _ = abc2dq0(i_vsc_a, i_vsc_b, i_vsc_c, angle)
        v_sh_d, v_sh_q, _ = abc2dq0(v_sh_a, v_sh_b, v_sh_c, angle)
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, angle)

        # Add delta_u of output feedback control
        delta_p_ref, delta_q_ref, delta_v_ref = self.output_feedback_control_emt(w, i_vsc_d, i_vsc_q, i_bus_d, i_bus_q)
        p_ref += delta_p_ref
        q_ref += delta_q_ref
        v_ref += delta_v_ref

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

        # Return derivatives of the state variables
        return d_vi + d_vd + d_vc + d_cc + d_lcl

    def plot_results_emt(self):
        """
        Plot EMT simulation results
        """

        angle, w, \
        q_f, \
        z_vc_d, z_vc_q, \
        z_cc_d, z_cc_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value        
        tps = self.variables_emt.x.time

        # Transform abc to dq0
        i_vsc_d, i_vsc_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, angle)])
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, angle)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, angle)])

        # Compute power
        p_sh = [v_d * i_d + v_q * i_q for v_d, v_q, i_d, i_q in zip(v_sh_d, v_sh_q, i_bus_d, i_bus_q)]
        q_sh = [v_q * i_d - v_d * i_q for v_d, v_q, i_d, i_q in zip(v_sh_d, v_sh_q, i_bus_d, i_bus_q)]

        results = DynamicalVariables(
            name=["angle", "w", "q_f", "z_vc_d", "z_vc_q", "z_cc_d", "z_cc_q", "i_vsc_d", "i_vsc_q", 
                  "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q", "p_sh", "q_sh"],
            component=f"{self.type_}_{self.id}",
            value=[angle, w, q_f, z_vc_d, z_vc_q, z_cc_d, z_cc_q, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q,
                    p_sh, q_sh],
            time=tps
        )
        return results

    def get_output_emt(self):
        
        angle, w, \
        q_f, \
        z_vc_d, z_vc_q, \
        z_cc_d, z_cc_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value      

        return [i_bus_a, i_bus_b, i_bus_c]

    