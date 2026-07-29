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
from sting.utils.transformations import R_DQ2dq, R_dq2DQ, d_DQ2dq_dangle, d_dq2DQ_dangle

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
        self.active_power_controller = ActivePowerPI1A(kp_pu=self.kp_pc_pu, ki_puHz=self.ki_pc_puHz)
        self.reactive_power_controller = ReactivePowerPI1A(kp_pu=self.kp_pc_pu, ki_puHz=self.ki_pc_puHz)

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
        # Unpack OPF solutions
        v_mag, phase_deg = self.power_flow_variables.vmag_bus, self.power_flow_variables.vphase_bus
        p_bus, q_bus = self.power_flow_variables.p_bus, self.power_flow_variables.q_bus
        # Compute initial conditions in the LCL filter
        lcl_init = self.lcl_filter.get_steady_state(
            v_bus_mag=v_mag, relative_phase_deg=phase_deg, p_bus=p_bus, q_bus=q_bus, reference_node = 'bus')
        # Unpack initial conditions
        i_bus_d, i_bus_q = lcl_init.i_bus_d, lcl_init.i_bus_q
        v_bus_d, v_bus_q = lcl_init.v_bus_d, lcl_init.v_bus_q
        v_vsc_d, v_vsc_q = lcl_init.v_vsc_d, lcl_init.v_vsc_q
        # PLL
        self.phase_locked_loop.get_steady_state(v_mag=v_mag, relative_phase_deg=phase_deg)        
        # Power controllers
        self.active_power_controller.get_steady_state(p_ref=p_bus, i_ref_d=i_bus_d)
        self.reactive_power_controller.get_steady_state(q_ref=q_bus, i_ref_q=i_bus_q)
        # Current controller
        self.current_controller.get_steady_state(
            v_out_d=v_vsc_d, v_out_q=v_vsc_q, v_d=v_bus_d, v_q=v_bus_q, i_d=i_bus_d, i_q=i_bus_q, w=1)

    
    def _build_small_signal_model(self):
        # Unpack OPF solutions
        v_mag, phase_deg = self.power_flow_variables.vmag_bus, self.power_flow_variables.vphase_bus
        p_bus, q_bus = self.power_flow_variables.p_bus, self.power_flow_variables.q_bus
        # Initial conditions in the LCL filter
        i_bus_d, i_bus_q = self.lcl_filter.emt_init.i_bus_d, self.lcl_filter.emt_init.i_bus_q
        i_vsc_d, i_vsc_q = self.lcl_filter.emt_init.i_vsc_d, self.lcl_filter.emt_init.i_vsc_q
        v_sh_d, v_sh_q = self.lcl_filter.emt_init.v_sh_d, self.lcl_filter.emt_init.v_sh_q
        v_bus_d, v_bus_q = self.lcl_filter.emt_init.v_bus_d, self.lcl_filter.emt_init.v_bus_q

        z_cc_d, z_cc_q = self.current_controller.emt_init.z_cc_d, self.current_controller.emt_init.z_cc_q

        # Create each components small-signal model
        pll_ssm = self.phase_locked_loop.get_small_signal_model(
            v_mag=v_mag, relative_phase_deg=phase_deg)
        apc_ssm = self.active_power_controller.get_small_signal_model(
            z_apc=i_bus_d, p_ref=p_bus, i_d=i_bus_d, i_q=i_bus_q, v_d=v_bus_d, v_q=v_bus_q)
        rpc_ssm = self.reactive_power_controller.get_small_signal_model(
            z_rpc=i_bus_q, q_ref=q_bus, i_d=i_bus_d, i_q=i_bus_q, v_d=v_bus_d, v_q=v_bus_q)
        cc_ssm = self.current_controller.get_small_signal_model(
            z_cc_d=z_cc_d, z_cc_q=z_cc_q, i_d=i_bus_d, i_q=i_bus_q, v_d=v_bus_d, v_q=v_bus_q, w=1
            )
        lcl_ssm = self.lcl_filter.get_small_signal_model(
            i_vsc_d=i_vsc_d, i_vsc_q=i_vsc_q, i_bus_d=i_bus_d, i_bus_q=i_bus_q, v_sh_d=v_sh_d, v_sh_q=v_sh_q)

        # Inverter level inputs and outputs
        v_bus_D, v_bus_Q = self.lcl_filter.emt_init.v_bus_D, self.lcl_filter.emt_init.v_bus_Q
        i_bus_D, i_bus_Q = self.lcl_filter.emt_init.i_bus_D, self.lcl_filter.emt_init.i_bus_Q
        u = DynamicalVariables(
            name=["p_ref", "q_ref", "v_bus_D", "v_bus_Q"],
            type=["device", "device", "grid", "grid"],
            init=[p_bus, q_bus, v_bus_D, v_bus_Q])
        y = DynamicalVariables(
            name=['i_bus_D', 'i_bus_Q'],
            init=[i_bus_D, i_bus_Q])

        # Generate small-signal model
        components = [pll_ssm, apc_ssm, rpc_ssm, cc_ssm, lcl_ssm]
        connections = self.get_interconnections_ssm(v_bus_D, v_bus_Q, i_bus_d, i_bus_q, phase_deg)
        self.ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        return self.ssm


    def get_interconnections_ssm(self, v_bus_D, v_bus_Q, i_bus_d, i_bus_q, relative_phase_deg):
        """
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


        ┌ component ──▶           │ PLL    │ APC      │ RPC      │ CC        │ LCL                            │ Grid inputs
        │       ┌ index ──▶       │ 0   1  │ 2        │ 3        │ 4,5       │ 6,7        8,9        10,11    │ 0       1       2,3
        ▼       ▼                 │ Δω  Δϕ │ Δi_ref_d │ Δi_ref_q │ Δv_vsc_dq │ Δi_vsc_dq  Δi_bus_dq  Δv_sh_dq │ Δp_ref  Δq_ref  Δv_bus_DQ
        ──────────────────────────┼────────┴──────────┴──────────┴───────────┴────────────────────────────────┼────────────────────────────
        PLL     0,1     Δv_bus_DQ │  0  0    0          0          0           0          0          0        │ 0       0       I₂
        APC     2       Δp_ref    │  0  0    0          0          0           0          0          0        │ 1       0       0
                3,4     Δi_bus_dq │  0  0    0          0          0           0          I₂         0        │ 0       0       0
                5,6     Δv_bus_dq │  0  a    0          0          0           0          0          0        │ 0       0       Rᵀ
        RPC     7       Δq_ref    │  0  0    0          0          0           0          0          0        │ 0       1       0
                8,9     Δi_bus_dq │  0  0    0          0          0           0          I₂         0        │ 0       0       0
                10,11   Δv_bus_dq │  0  a    0          0          0           0          0          0        │ 0       0       Rᵀ
        CC      12      Δi_ref_d  │  0  0    1          0          0           0          0          0        │ 0       0       0
                13      Δi_ref_q  │  0  0    0          1          0           0          0          0        │ 0       0       0
                14,15   Δi_bus_dq │  0  0    0          0          0           0          I₂         0        │ 0       0       0
                16,17   Δv_bus_dq │  0  a    0          0          0           0          0          0        │ 0       0       Rᵀ
                18      Δw        │  1  0    0          0          0           0          0          0        │ 0       0       0
        LCL     19,20   Δv_vsc_dq │  0  0    0          0          I₂          0          0          0        │ 0       0       0
                21,22   Δv_bus_dq │  0  a    0          0          0           0          0          0        │ 0       0       Rᵀ
                23      Δw        │  1  0    0          0          0           0          0          0        │ 0       0       0
        ──────────────────────────┼───────────────────────────────────────────────────────────────────────────┼────────────────────────────
        Grid    0,1     Δi_bus_DQ │  0  b    0          0          0           0          R          0        │ 0       0       0
        outputs                  
        """

        angle = relative_phase_deg * np.pi / 180
        R = R_dq2DQ(angle)
        I = np.eye(2)

        a = d_DQ2dq_dangle(v_bus_D, v_bus_Q, angle).reshape(2,1)
        b = d_dq2DQ_dangle(i_bus_d, i_bus_q, angle).reshape(2,1)

        F = np.zeros((24, 12))
        G = np.zeros((24, 4))
        H = np.zeros((2, 12))
        L = np.zeros((2, 4))

        # Entries in F and G entered as tuples: (row_idx, col_idx, values)
        idx_F =[
            ([3,4], [8,9], I), ([8,9], [8,9], I), ([12,13], [2,3], I), ([14,15], [8,9], I), ([19,20], [4,5], I),
            ([5,6], [1], a), ([10,11], [1], a), ([16,17], [1], a), ([21,22], [1], a), ([18], [0], 1), ([23], [0], 1)
            ]
        for rows, cols, value in idx_F:
            F[np.ix_(rows, cols)] = value
        
        idx_G = [
            ([2], [0], 1), ([7], [1], 1), ([0,1], [2,3], I), 
            ([5,6], [2,3], R.T), ([10,11], [2,3], R.T), ([16,17], [2,3], R.T), ([21,22], [2,3], R.T)
            ]
        for rows, cols, value in idx_G:
            G[np.ix_(rows, cols)] = value
        # Add values to H
        H[:,[1]] = b
        H[np.ix_([0,1],[8,9])] = R

        return (F,G,H,L) 
        

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
                self.active_power_controller.emt_init.z_apc,
                self.reactive_power_controller.emt_init.z_rpc,
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
        (v_pll_q, z_pll, theta_pll, z_apc, z_rpc, z_cc_d, z_cc_q,
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
        d_z_apc = self.active_power_controller.get_derivatives_step_emt_abc(p_ref=p_ref, p=p_bus, z_apc=z_apc)
        d_z_rpc = self.reactive_power_controller.get_derivatives_step_emt_abc(q_ref=q_ref, q=q_bus, z_rpc=z_rpc)
        # Reference currents from power controller
        i_ref_d = self.active_power_controller.get_algebraics_step_emt_abc(p_ref=p_ref, p=p_bus, z_apc=z_apc)
        i_ref_q = self.reactive_power_controller.get_algebraics_step_emt_abc(q_ref=q_ref, q=q_bus, z_rpc=z_rpc)

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


    def get_output_emt(self):
            (v_pll_q, z_pll, theta_pll, z_apc, z_rpc, z_cc_d, z_cc_q,
                    i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c) = self.variables_emt.x.value
                
            return [i_bus_a, i_bus_b, i_bus_c]


    def plot_results_emt(self):
        """
        Plot EMT simulation results
        """
        (v_pll_q, z_pll, theta_pll, z_apc, z_rpc, z_cc_d, z_cc_q,
                i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c) = self.variables_emt.x.value

        # Transform abc to dq0
        i_vsc_d, i_vsc_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, theta_pll)])
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, theta_pll)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, theta_pll)])
        

        results = DynamicalVariables(
            name = ['v_pll_q', 'z_pll', 'theta_pll', 'z_apc', 'z_rpc',  'z_cc_d', 'z_cc_q', "i_vsc_d", "i_vsc_q", "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q"],
            component = f"{self.type_}_{self.id}",
            value=[v_pll_q, z_pll, theta_pll, z_apc, z_rpc, z_cc_d, z_cc_q, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q],
            time=self.variables_emt.x.time
        )

        return results