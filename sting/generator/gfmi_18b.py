"""
This module implements a 18th order Grid-Forming Inverter comprised of: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Current controller: A dq-based frame PI controller
- Voltage controller: A dq-based frame PI controller
- Virtual inertia (active power control): A virtual inertia control that emulates the behavior of a synchronous generator.
- Voltage droop (reactive power control): A droop control that emulates the behavior of a synchronous generator.
"""
from dataclasses import dataclass, field

import numpy as np

from sting.components import (
    InnerCurrentController2B,
    InnerVoltageController2B,
    LCLFilter9A,
    ParallelRCShunt2A,
    SeriesRLBranch2A,
    SeriesRLBranch2B,
    RotationalInertia2A,
    VoltageDroopController1A,
)
from sting.generator.core import Generator
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.dynamical_systems import (
    DynamicalVariables,
    QuadraticBilinearModel,
    StateSpaceModel,
)
from sting.utils.transformations import (
    R_DQ2dq,
    R_dq2DQ,
    abc2dq0,
    d_DQ2dq_dangle,
    d_dq2DQ_dangle,
    dq02abc,
)


@dataclass(slots=True, kw_only=True, eq=False)
class GFMI18B(Generator):
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
    alpha: float
    # Voltage droop parameters
    k_q_pu: float
    w_q_puHz: float

    # Components
    lcl_filter: LCLFilter9A = field(init=False)
    # LCL filter components for quadratic bilinear model
    lcl_br1: SeriesRLBranch2B = field(init=False)
    lcl_br2: SeriesRLBranch2A = field(init=False)
    lcl_sh: ParallelRCShunt2A  = field(init=False)
    
    voltage_controller: InnerVoltageController2B = field(init=False)
    current_controller: InnerCurrentController2B = field(init=False)
    virtual_inertia: RotationalInertia2A = field(init=False)
    voltage_droop: VoltageDroopController1A = field(init=False)


    def __post_init__(self):
        self.lcl_filter = LCLFilter9A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.lcl_br1 = SeriesRLBranch2B(self.rf1_pu, self.xf1_pu, self.wbase)
        self.lcl_br2 = SeriesRLBranch2A(self.rf2_pu, self.xf2_pu, self.wbase)
        self.lcl_sh = ParallelRCShunt2A(1/self.rsh_pu, self.csh_pu, self.wbase)
        self.voltage_controller = InnerVoltageController2B(self.kp_vc_pu, self.ki_vc_puHz, self.kffi_vc, self.csh_pu)
        self.current_controller = InnerCurrentController2B(self.kp_cc_pu, self.ki_cc_puHz, self.kffv_cc, self.xf1_pu)
        self.virtual_inertia = RotationalInertia2A(self.h_s, self.kd_pu, self.wbase, self.alpha)
        self.voltage_droop = VoltageDroopController1A(self.k_q_pu, self.w_q_puHz)

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
       )

        self.current_controller.get_steady_state(
           v_out_d=lcl_init.v_vsc_d,
           v_out_q=lcl_init.v_vsc_q,
           v_d=lcl_init.v_sh_d,
           v_q=lcl_init.v_sh_q,
           i_d=lcl_init.i_vsc_d,
           i_q=lcl_init.i_vsc_q
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

    def get_derivative_state_emt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:

        # Extract states
        angle, w, \
        q_f, \
        z_vc_d, z_vc_q, \
        z_cc_d, z_cc_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = x

        # Get inputs
        p_ref, q_ref, v_ref, v_bus_a, v_bus_b, v_bus_c = u

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
        i_ref_d, i_ref_q = self.voltage_controller.get_algebraics_step_emt_dq0(z_vc_d, z_vc_q, u_ref_d, u_ref_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q)

        # Compute voltage reference for the LCL filter
        v_vsc_d, v_vsc_q = self.current_controller.get_algebraics_step_emt_dq0(z_cc_d, z_cc_q, i_ref_d, i_ref_q, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q)

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

    def get_output_emt(self, x: np.ndarray) -> np.ndarray:
        
        angle, w, \
        q_f, \
        z_vc_d, z_vc_q, \
        z_cc_d, z_cc_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = x     

        return [i_bus_a, i_bus_b, i_bus_c]


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
        )

        current_controller_ssm = self.current_controller.get_small_signal_model(
            z_cc_d = self.current_controller.emt_init.z_cc_d,
            z_cc_q = self.current_controller.emt_init.z_cc_q,
            i_d = self.lcl_filter.emt_init.i_vsc_d,
            i_q = self.lcl_filter.emt_init.i_vsc_q,
            v_d = self.lcl_filter.emt_init.v_sh_d,
            v_q = self.lcl_filter.emt_init.v_sh_q
        )

        lcl_filter_ssm = self.lcl_filter.get_small_signal_model(
            i_vsc_d = self.lcl_filter.emt_init.i_vsc_d,
            i_vsc_q = self.lcl_filter.emt_init.i_vsc_q,
            i_bus_d = self.lcl_filter.emt_init.i_bus_d,
            i_bus_q = self.lcl_filter.emt_init.i_bus_q,
            v_sh_d = self.lcl_filter.emt_init.v_sh_d,
            v_sh_q = self.lcl_filter.emt_init.v_sh_q
        )

        # Inputs and outputs
        u = DynamicalVariables(
            name=["p_ref", "q_ref", "v_ref", "v_bus_D", "v_bus_Q"],
            type=["device", "device", "device", "grid", "grid"],
            init=[self.virtual_inertia.emt_init.p_ref,
                  self.voltage_droop.emt_init.q_ref,
                  self.voltage_droop.emt_init.v_ref,
                  self.lcl_filter.emt_init.v_bus_D,
                  self.lcl_filter.emt_init.v_bus_Q])

        y = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            init=[self.lcl_filter.emt_init.i_bus_D, 
                  self.lcl_filter.emt_init.i_bus_Q])

        # Generate small-signal model
        components = [virtual_inertia_ssm, voltage_droop_ssm, voltage_controller_ssm, current_controller_ssm, lcl_filter_ssm]
        connections = self.get_interconnections_ssm(self.lcl_filter.emt_init.v_bus_D, 
                                                    self.lcl_filter.emt_init.v_bus_Q,
                                                    self.lcl_filter.emt_init.i_bus_d, 
                                                    self.lcl_filter.emt_init.i_bus_q,
                                                    self.lcl_filter.emt_init.angle_ref)
        self.ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        return self.ssm

    def get_interconnections_ssm(self, v_bus_D, v_bus_Q, i_bus_d, i_bus_q, relative_phase_rad):
        """
        Construct the interconnection matrices F, H, G, and L that satisfies:
        u_stack = F * y_stack + H * u_sys
        y_sys   = G * y_stack + L * u_sys

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

        ┌ component ──▶             |   APC    ┆ RPC        ┆ IVC        ┆ ICC        ┆ LCL                            │ Grid inputs
        │       ┌ index ──▶         │   0   1  ┆ 2,3        ┆ 4,5        ┆ 6,7        ┆ 8,9        10,11      12,13    │ 0       1       2       3,4 
        ▼       ▼                   │   Δϕ  Δω ┆ Δv_ref_dq  ┆ Δi_ref_dq  ┆ Δv_vsc_dq  ┆ Δi_vsc_dq  Δi_bus_dq  Δv_sh_dq │ Δp_ref  Δq_ref  Δv_ref  Δv_bus_DQ
        ────────────────────────────┼──────────┴────────────┴────────────┴────────────┴────────────────────────────────┼─────────────────────────────────────
        APC     0       Δp_ref      │   0   0    0            0            0            0          0          0        │   1        0        0        0     
                1,2     Δi_bus_dq   │   0   0    0            0            0            0          I₂         0        │   0        0        0        0     
                3,4     Δv_sh_dq    │   0   0    0            0            0            0          0          I₂       │   0        0        0        0     
        RPC     5       Δq_ref      │   0   0    0            0            0            0          0          0        │   0        1        0        0     
                6       Δv_ref      │   0   0    0            0            0            0          0          0        │   0        0        1        0     
                7,8     Δi_bus_dq   │   0   0    0            0            0            0          I₂         0        │   0        0        0        0     
                9,10    Δv_sh_dq    │   0   0    0            0            0            0          0          I₂       │   0        0        0        0     
        IVC     11,12   Δv_ref_dq   │   0   0    I₂           0            0            0          0          0        │   0        0        0        0     
                13,14   Δv_sh_dq    │   0   0    0            0            0            0          0          I₂       │   0        0        0        0     
                15,16   Δi_bus_dq   │   0   0    0            0            0            0          I₂         0        │   0        0        0        0        
        ICC     17,18   Δi_ref_dq   │   0   0    0            I₂           0            0          0          0        │   0        0        0        0     
                19,20   Δi_vsc_dq   │   0   0    0            0            0            I₂         0          0        │   0        0        0        0     
                21,22   Δv_sh_dq    │   0   0    0            0            0            0          0          I₂       │   0        0        0        0        
        LCL     23,24   Δv_vsc_dq   │   0   0    0            0            I₂           0          0          0        │   0        0        0        0     
                25,26   Δv_bus_dq   │   a   0    0            0            0            0          0          0        │   0        0        0        Rᵀ     
                27      Δω          │   0   1    0            0            0            0          0          0        │   0        0        0        0  
        ────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────
        Grid    0,1     Δi_bus_DQ   │   b   0    0            0            0            0          R          0        │   0        0        0        0
        outputs
        """
        angle = relative_phase_rad
        R = R_dq2DQ(angle)
        I = np.eye(2)

        a = d_DQ2dq_dangle(v_bus_D, v_bus_Q, angle).reshape(2,1)
        b = d_dq2DQ_dangle(i_bus_d, i_bus_q, angle).reshape(2,1)

        F = np.zeros((28, 14))
        G = np.zeros((28, 5))
        H = np.zeros((2, 14))
        L = np.zeros((2, 5))

        idx_F =[
            ([1,2], [10,11], I), ([3,4], [12,13], I), ([7,8], [10,11], I), ([9,10], [12,13], I),
            ([11,12], [2,3], I), ([13,14], [12,13], I), ([15,16], [10,11], I), ([17], [1], 1),
            ([17,18], [4,5], I), ([19,20], [8,9], I), ([21,22], [12,13], I),  ([24], [1], 1), 
            ([23,24], [6,7], I), ([25,26], [0], a), ([27], [1], 1), 
        ]
        for rows, cols, value in idx_F:
            F[np.ix_(rows, cols)] = value

        idx_G = [
            ([0], [0], 1), ([5,6],[1,2], I), ([25,26], [3,4], R.T)
        ]
        for rows, cols, value in idx_G:
            G[np.ix_(rows, cols)] = value
        
        H[:, [0]] = b
        H[np.ix_([0,1], [10,11])] = R

        return F, G, H, L



    def _build_quadratic_bilinear_model(self):
        init =  self.lcl_filter.emt_init
        
        # Power controllers
        apc_qbm = self.virtual_inertia.get_quadratic_bilinear_model(
            w = 1,
            angle_rad = self.virtual_inertia.emt_init.angle,
            p_ref = self.virtual_inertia.emt_init.p_ref,
            p = self.virtual_inertia.emt_init.p_ref,
        )
        rpc_qbm = self.voltage_droop.get_quadratic_bilinear_model(
            q_ref = self.voltage_droop.emt_init.q_ref,
            v_ref = self.voltage_droop.emt_init.v_ref,
            q = self.voltage_droop.emt_init.q_ref
        )
        # Inner-controllers
        ivc_qbm = self.voltage_controller.get_quadratic_bilinear_model(
            z_vc_d = self.voltage_controller.emt_init.z_vc_d,
            z_vc_q = self.voltage_controller.emt_init.z_vc_q,
            v_d = init.v_sh_d,
            v_q = init.v_sh_q,
            i_d = init.i_bus_d,
            i_q = init.i_bus_q,
        )
        icc_qbm = self.current_controller.get_quadratic_bilinear_model(
            z_cc_d = self.current_controller.emt_init.z_cc_d,
            z_cc_q = self.current_controller.emt_init.z_cc_q,
            i_d = init.i_vsc_d,
            i_q = init.i_vsc_q,
            v_d = init.v_sh_d,
            v_q = init.v_sh_q
        )
        # LCL Filter
        br1_qbm = self.lcl_br1.get_quadratic_bilinear_model(
            v_from_d = init.v_vsc_d, 
            v_from_q = init.v_vsc_q, 
            v_to_d = init.v_sh_d, 
            v_to_q = init.v_vsc_q,
            i_d = init.i_vsc_d, 
            i_q = init.i_vsc_q
            )
        br2_qbm = self.lcl_br2.get_quadratic_bilinear_model(
            v_from_D = init.v_sh_D, 
            v_from_Q = init.v_sh_Q,
            v_to_D = init.v_bus_D, 
            v_to_Q = init.v_bus_Q,
            i_D = init.i_bus_D, 
            i_Q = init.i_bus_Q
        )
        sh_qbm = self.lcl_sh.get_quadratic_bilinear_model(
            v_D = init.v_sh_D, 
            v_Q = init.v_sh_Q, 
            i_D = (init.i_vsc_D - init.i_bus_D), 
            i_Q = (init.i_vsc_Q - init.i_bus_Q) 
        )

        # Inputs and outputs
        u = DynamicalVariables(
            name=["p_ref",  "q_ref",  "v_ref",  "one",   "v_bus_D", "v_bus_Q"],
            type=["device", "device", "device", "device", "grid",   "grid"],
            init=[
                self.virtual_inertia.emt_init.p_ref,
                self.voltage_droop.emt_init.q_ref,
                self.voltage_droop.emt_init.v_ref,
                1,
                init.v_bus_D,
                init.v_bus_Q
            ])

        y = DynamicalVariables(name=["i_bus_D", "i_bus_Q"], init=[init.i_bus_D, init.i_bus_Q])


        # Generate quadratic bilinear model
        components = [apc_qbm, rpc_qbm, ivc_qbm, icc_qbm, br1_qbm, br2_qbm, sh_qbm]
        connections = self.get_interconnections_qbm()
        self.qbm = QuadraticBilinearModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        return self.qbm

    
    def get_interconnections_qbm(self):
        """        
        LINEAR INTERCONNECTIONS
        -----------------------
        ┌ component ──▶            | VI APC      ┆ RPC      ┆ IVC       ┆ ICC       ┆ RL_1      RL_2      RC      │ Grid inputs
        │       ┌ index ──▶        │ 0   1   2   ┆ 3,4      ┆ 5,6       ┆ 7,8       ┆ 9,10      11,12     13,14   │ 0      1      2      3    4,5 
        ▼       ▼                  │ ω   sin cos ┆ v_ref_dq ┆ i_ref_dq  ┆ v_vsc_dq  ┆ i_vsc_dq  i_bus_DQ  v_sh_DQ │ p_ref  q_ref  v_ref  one  v_bus_DQ
        ───────────────────────────┼─────────────┴──────────┴───────────┴───────────┴─────────────────────────────┼─────────────────────────────────────
        APC     0        p_ref     │ 0   0   0     0          0           0           0         0         0       │ 1       0     0      0    0
                1        one       │ 0   0   0     0          0           0           0         0         0       │ 0       0     0      1    0
                2       *p_shunt   │ 0   0   0     0          0           0           0         0         0       │ 0       0     0      0    0
        RPC     3        q_ref     │ 0   0   0     0          0           0           0         0         0       │ 0       1     0      0    0
                4        v_ref     │ 0   0   0     0          0           0           0         0         0       │ 0       0     1      0    0
                5       *q_shunt   │ 0   0   0     0          0           0           0         0         0       │ 0       0     0      0    0
        IVC     6,7      v_ref_dq  │ 0   0   0     I₂         0           0           0         0         0       │ 0       0     0      0    0
                8,9     *v_sh_dq   │ 0   0   0     0          0           0           0         0         0       │ 0       0     0      0    0
                10,11   *i_bus_dq  │ 0   0   0     0          0           0           0         0         0       │ 0       0     0      0    0
        ICC     12,13    i_ref_dq  │ 0   0   0     0          I₂          0           0         0         0       │ 0       0     0      0    0
                14,15    i_vsc_dq  │ 0   0   0     0          0           0           I₂        0         0       │ 0       0     0      0    0
                16,17   *v_sh_dq   │ 0   0   0     0          0           0           0         0         0       │ 0       0     0      0    0
        RL1     18,19    v_vsc_dq  │ 0   0   0     0          0           I₂          0         0         0       │ 0       0     0      0    0
                20,21   *v_sh_dq   │ 0   0   0     0          0           0           0         0         0       │ 0       0     0      0    0
                22       ω         │ 1   0   0     0          0           0           0         0         0       │ 0       0     0      0    0
        RL2     23,24    v_sh_DQ   │ 0   0   0     0          0           0           0         0         I₂      │ 0       0     0      0    0
                25,26    v_bus_DQ  │ 0   0   0     0          0           0           0         0         0       │ 0       0     0      0    I₂
        RC      27,28   *i_sh_DQ   │ 0   0   0     0          0           0           0        -I₂        0       │ 0       0     0      0    0
        ───────────────────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────
        Grid    0,1      i_bus_DQ  │ 0   0   0     0          0           0           0         I₂        0       │ 0       0     0      0    0
        outputs

        idx_11 = [([6,7],[3,4],I), ([12,13],[5,6],I), ([14,15],[9,10],I), ([18,19],[7,8],I), ([22],[0],1) ([23,24],[13,14],I),([27,28],[11,12],-I)]
        idx_12 = [([0],[0],1), ([1],[3],1), ([3,4],[2,3],I), ([25,26],[4,5],I)]

        NONLINEAR INTERCONNECTIONS
        --------------------------
        Recall the transformation from DQ to dq  
            i_d =  i_D*cos + i_Q*sin
            i_q = -i_D*sin + i_Q*cos
        
        Active and reactive power
            p = v_d * i_d + v_q * i_q
            q = v_q * i_d - v_d * i_q
        
        We will define
            J = [ 0  1]
                [-1  0]
            
                                   | VI APC      ┆ RPC ┆ IVC/ICC ┆ RL_1      RL_2      RC     
                             1     │ 0   1   2   ┆ 3   ┆ 4,5,6,7 ┆ 8,9       10,11     12,13    
        (x_1 * x)            sin * │ ω   sin cos ┆ q_f ┆ ...     ┆ i_vsc_dq  i_bus_DQ  v_sh_DQ
        ───────────────────────────┼─────────────┴─────┴─────────┴─────────────────────────────
        IVC     8,9     *v_sh_dq   │ 0   0   0     0     0         0         0         J₂       
                10,11   *i_bus_dq  │ 0   0   0     0     0         0         J₂        0 
        ICC     16,17   *v_sh_dq   │ 0   0   0     0     0         0         0         J₂
        RL1     20,21   *v_sh_dq   │ 0   0   0     0     0         0         0         J₂
        RC      27,28   *i_sh_DQ   │ 0   0   0     0     0        -J₂        0         0 

        idx_x1 = [([8,9],[12,13],J), ([10,11],[10,11],J), ([16,17],[12,13],J), ([20,21],[12,13],J), ([27,28],[8,9],J.T)]
        
                             2     │ 0   1   2   ┆ 3   ┆ 4,5,6,7 ┆ 8,9       10,11     12,13    
        (x_2 * x)            cos * │ ω   sin cos ┆ q_f ┆ ...     ┆ i_vsc_dq  i_bus_DQ  v_sh_DQ
        ───────────────────────────┼─────────────┴─────┴─────────┴─────────────────────────────
        IVC     8,9     *v_sh_dq   │ 0   0   0     0     0         0         0         I₂         
                10,11   *i_bus_dq  │ 0   0   0     0     0         0         I₂        0 
        ICC     16,17   *v_sh_dq   │ 0   0   0     0     0         0         0         I₂ 
        RL1     20,21   *v_sh_dq   │ 0   0   0     0     0         0         0         I₂
        RC      27,28   *i_sh_DQ   │ 0   0   0     0     0         I₂        0         0 

        idx_x2 = [([8,9],[12,13],I), ([10,11],[10,11],I), ([16,17],[12,13],I), ([20,21],[12,13],I), ([27,28],[8,9],I)]


                                   | VI APC      ┆ RPC ┆ IVC/ICC ┆ RL_1      RL_2              RC     
                          12       │ 0   1   2   ┆ 3   ┆ 4,5,6,7 ┆ 8,9       10       11       12,13    
        (x_12 * x)        v_sh_D * │ ω   sin cos ┆ q_f ┆ ...     ┆ i_vsc_dq  i_bus_D  i_bus_Q  v_sh_DQ
        ───────────────────────────┼─────────────┴─────┴─────────┴─────────────────────────────────────
        APC     2       *p_shunt   │ 0   0   0     0     0         0         1        0        0 
        RPC     5       *q_shunt   │ 0   0   0     0     0         0         0       -1        0 
 
        idx_x12 = [([2],[10],1),([5],[11],-1)]

                          13       │ 0   1   2   ┆ 3   ┆ 4,5,6,7 ┆ 8,9       10       11       12,13    
        (x_13 * x)        v_sh_Q * │ ω   sin cos ┆ q_f ┆ ...     ┆ i_vsc_dq  i_bus_D  i_bus_Q  v_sh_DQ
        ───────────────────────────┼─────────────┴─────┴─────────┴─────────────────────────────────────
        APC     2       *p_shunt   │ 0   0   0     0     0         0         0        1        0 
        RPC     5       *q_shunt   │ 0   0   0     0     0         0         1        0        0 

        idx_x13 = [([2],[11],1), ([5],[10],1)]
        """
        # Common matrices
        I = np.eye(2)
        J = np.array([[0, 1], [-1,0]])

        # Number of stacked/grid side inputs and outputs
        u_stack = 29
        y_stack = 15
        x_stack = 14
        u_grid = 6
        y_grid = 2

        # Linear interconnection matrices
        L11 = np.zeros((u_stack, y_stack))
        L12 = np.zeros((u_stack, u_grid))
        L21 = np.zeros((y_grid, y_stack))
        L22 = np.zeros((y_grid, u_grid))

        idx_11 = [([6,7],[3,4],I), ([12,13],[5,6],I), ([14,15],[9,10],I), ([18,19],[7,8],I), ([22],[0],1), ([23,24],[13,14],I),([27,28],[11,12],-I)]
        idx_12 = [([0],[0],1), ([1],[3],1), ([3,4],[1,2],I), ([25,26],[4,5],I)]
        idx_21 = [([0,1],[11,12], I)]

        # Nonlinear interconnection matrices
        M1_x1, M1_x2, M1_x12, M1_x13 = (np.zeros((u_stack, x_stack)) for _ in range(4))
        M2 = np.zeros((u_stack, x_stack*u_grid))
        
        idx_x1 = [([8,9],[12,13],J), ([10,11],[10,11],J), ([16,17],[12,13],J), ([20,21],[12,13],J), ([27,28],[8,9],J.T)]
        idx_x2 = [([8,9],[12,13],I), ([10,11],[10,11],I), ([16,17],[12,13],I), ([20,21],[12,13],I), ([27,28],[8,9],I)]
        idx_x12 = [([2],[10],1),([5],[11],-1)]
        idx_x13 = [([2],[11],1), ([5],[10],1)]

        # Fill out each matrix
        matrix_index_pairs =  [(L11, idx_11), (L12, idx_12), (L21, idx_21), (M1_x1, idx_x1), (M1_x2, idx_x2), (M1_x12, idx_x12), (M1_x13, idx_x13)]
        for matrix, idx in matrix_index_pairs:
            for rows, cols, value in idx:
                matrix[np.ix_(rows, cols)] = value

        # Stack matrices in M1
        M1 = np.hstack((np.zeros((u_stack, x_stack)), M1_x1, M1_x2, np.zeros((u_stack, 9*x_stack)), M1_x12, M1_x13))
                      
        return (L11, L12, L21, L22, M1, M2)