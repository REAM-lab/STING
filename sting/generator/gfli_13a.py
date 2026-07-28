"""
This module implements a 13th order Grid-following Inverter (GFLI) comprised of:
- 3rd order PLL with filter: It that tracks the phase of the grid voltage.
- 1st order active power controller: A PI controller that regulates the active power of the inverter.
- 1st order reactive power controller: A PI controller that regulates the reactive power of the inverter.
- 2nd order current controller: A dq-based frame PI controller
- 6th order LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
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
from sting.components import PhaseLockedLoop3A, InnerCurrentController2A, LCLFilter6A, ActivePowerPI1A, ReactivePowerPI1A

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
    tau_pll: float
    # Current controller parameters
    kp_cc_pu: float
    ki_cc_puHz: float
    kff_cc: float
    # Power controller parameters
    kp_pc_pu: float
    ki_pc_puHz: float

    # Components
    lcl_filter: LCLFilter6A = field(init=False)
    current_controller: InnerCurrentController2A = field(init=False)
    phase_locked_loop: PhaseLockedLoop3A = field(init=False)
    active_power_controller: ActivePowerPI1A = field(init=False)
    reactive_power_controller: ReactivePowerPI1A = field(init=False)

    def __post_init__(self):
        self.lcl_filter = LCLFilter6A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.phase_locked_loop = PhaseLockedLoop3A(self.kp_pll_pu, self.ki_pll_puHz, self.tau_pll, self.wbase)
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
       
        """    lcl_init = self.lcl_filter.get_steady_state(
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
                v_bus_mag = self.power_flow_variables.vmag_bus,
                relative_phase_deg = self.power_flow_variables.vphase_bus
            )"""

    
    def _build_small_signal_model(self):
        pass


    def get_interconnections_ssm(self, v_bus_D, v_bus_Q, i_bus_d, i_bus_q, relative_phase_deg):
        pass


    def define_variables_emt(self):
        # States 
        x = DynamicalVariables(
            name = [
                'v_pll_q', 'z_pll', 'theta_pll', 'z_apc', 'z_rpc',  'z_cc_d', 'z_cc_q',
                "i_vsc_a", "i_vsc_b", "i_vsc_c", "v_sh_a", "v_sh_b","v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c"],
            component = f"{self.type_}_{self.id}",
            init = [
                # PLL
                self.phase_locked_loop.emt_init.v_pll_q,
                self.phase_locked_loop.emt_init.z_pll,
                self.phase_locked_loop.emt_init.theta_pll, 
                # Power control
                self.active_power_controller.emt_init.z_pi,
                self.reactive_power_controller.emt_init.z_pi,
                # Current control
                self.current_controller.emt_init.z_cc_d, 
                self.current_controller.emt_init.z_cc_q,
                # LCL
                self.lcl_filter.emt_init.i_vsc_a, self.lcl_filter.emt_init.i_vsc_b, self.lcl_filter.emt_init.i_vsc_c,
                self.lcl_filter.emt_init.v_sh_a, self.lcl_filter.emt_init.v_sh_b, self.lcl_filter.emt_init.v_sh_c,
                self.lcl_filter.emt_init.i_bus_a, self.lcl_filter.emt_init.i_bus_b, self.lcl_filter.emt_init.i_bus_c]
        )

        # Inputs 
        u = DynamicalVariables(
            name=["p_ref", "q_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "grid", "grid", "grid"],
            init=[
                self.active_power_controller.emt_init.p_ref,
                self.reactive_power_controller.emt_init.q_ref,
                self.lcl_filter.emt_init.v_bus_a, 
                self.lcl_filter.emt_init.v_bus_b, 
                self.lcl_filter.emt_init.v_bus_c]
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[self.lcl_filter.emt_init.i_bus_a, self.lcl_filter.emt_init.i_bus_b, self.lcl_filter.emt_init.i_bus_c]
        )
        
        self.variables_emt = VariablesEMT(x=x,u=u,y=y)

    def get_derivative_state_emt(self):
        """
        It returns a vector with the differential equations that describe the dynamics of the GFLI.
        This model includes: pi controller, pll, and LCL filter.
        """     
        # Unpack states
        ( v_pll_q, z_pll, theta_pll, z_apc, z_rpc, z_cc_d, z_cc_q,
        i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c) = self.variables_emt.x.value
        # Unpack *external* inputs
        p_ref, q_ref, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        # Compute relevant quantities in the converter reference frame
        v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, theta_pll) 
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, theta_pll) 
        p_bus = v_bus_d * i_bus_d + v_bus_q * i_bus_q
        q_bus = v_bus_q * i_bus_d - v_bus_d * i_bus_q

        #### Phase-locked loop ####
        d_x_pll = self.phase_locked_loop.get_derivatives_step_emt_abc(
            v_pll_q, z_pll, theta_pll, v_a=v_bus_a, v_b=v_bus_b, v_c=v_bus_c)
        # Frequency estimated by PLL
        w_pll  = d_x_pll[2]/self.wbase

        #### Power controller ####
        d_z_apc = self.active_power_controller.get_derivatives_step_emt_abc(p_ref=p_ref, p=p_bus, z_pi=z_apc)
        d_z_rpc = self.reactive_power_controller.get_derivatives_step_emt_abc(q_ref=q_ref, q=q_bus, z_pi=z_rpc)
        # Reference currents from power controller
        i_ref_d = self.active_power_controller.get_algebraics_step_emt_abc(p_ref=p_ref, p=p_bus, z_pi=z_apc)
        i_ref_q = self.reactive_power_controller.get_algebraics_step_emt_abc(q_ref=q_ref, q=q_bus, z_pi=z_rpc)

        #### Current controller ####
        d_x_cc = self.current_controller.get_derivatives_step_emt_dq0(i_ref_d, i_ref_q, i_bus_d, i_bus_q)
        # Compute the voltage references from the inner current controller
        v_vsc_d, v_vsc_q = self.current_controller.get_algebraics_step_emt_dq0(
            z_cc_d, z_cc_q, i_ref_d, i_ref_q, i_bus_d, i_bus_q, v_bus_d, v_bus_q, w_pll)
        # Convert to abc to feed into filter dynamics 
        v_vsc_a, v_vsc_b, v_vsc_c = dq02abc(v_vsc_d, v_vsc_q, 0, theta_pll) 
        
        #### LCL filter ####
        d_x_lcl = self.lcl_filter.get_derivatives_step_emt_abc(
            i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c,
            v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c 
            )
        
        return d_x_pll + [d_z_apc, d_z_rpc] + d_x_cc + d_x_lcl