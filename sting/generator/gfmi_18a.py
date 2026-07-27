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
from sting.utils.transformations import dq02abc, abc2dq0
from sting.components import LCLFilter6A, InnerVoltageController2A, InnerCurrentController2A, VirtualInertia2A, VoltageDroopController1A

@dataclass(slots=True, kw_only=True, eq=False)
class GFMI18A(Generator):
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
    kffv_cc_pu: float
    # Inner voltage controller parameters
    kp_vc_pu: float
    ki_vc_puHz: float
    kffi_vc_pu: float
    # Virtual inertia parameters
    h_pu: float
    kd_pu: float
    # Voltage droop parameters
    k_q_pu: float
    w_q_puHz: float

    # Components
    lcl_filter: LCLFilter6A = field(init=False)
    voltage_controller: InnerVoltageController2A = field(init=False)
    current_controller: InnerCurrentController2A = field(init=False)
    virtual_inertia: VirtualInertia2A = field(init=False)
    voltage_droop: VoltageDroopController1A = field(init=False)


    def __post_init__(self):
        self.lcl_filter = LCLFilter6A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.voltage_controller = InnerVoltageController2A(self.kp_vc_pu, self.ki_vc_puHz, self.kffi_vc_pu, self.csh_pu)
        self.current_controller = InnerCurrentController2A(self.kp_cc_pu, self.ki_cc_puHz, self.kff_cc_pu, self.xf1_pu)
        self.virtual_inertia = VirtualInertia2A(self.h_pu, self.kd_pu)
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
        
    def define_variables_emt(self):
        # Initial conditions for the LCL filter
        init = self.lcl_filter.emt_init

        # States 
        # ------ 
        relative_phase_deg = np.atan(init.v_sh_q / init.v_sh_d) * np.pi / 180
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