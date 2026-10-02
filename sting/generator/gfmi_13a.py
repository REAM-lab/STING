"""
This module implements a voltage-controlled Grid-Forming Inverter (vcGFM).
"""

import numpy as np
from dataclasses import dataclass, field

from sting.branch.series_rl_branch_2a import SeriesRLBranch2A

from sting.components import (
    LCLFilter9A,
    RotationalInertia2A,
)

from sting.generator.core import Generator

from sting.modules.simulation_emt.utils import VariablesEMT

from sting.shunt.parallel_rc_shunt_2a import ParallelRCShunt2A

from sting.utils.dynamical_systems import (
    DynamicalVariables,
    QuadraticBilinearModel,
    StateSpaceModel,
)

from sting.utils.matrix_tools import coordinates_to_matrix

from sting.utils.transformations import (
    R_DQ2dq,
    R_dq2DQ,
    abc2dq0,
    d_DQ2dq_dangle,
    d_dq2DQ_dangle,
    dq02abc,
)

# ======================================================================
# vcGFM
# ======================================================================

@dataclass(slots=True, kw_only=True, eq=False)
class VCGFM(Generator):
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

    # Virtual inertia parameters
    h_s: float
    kd_pu: float
    alpha: float = 1

    # TVR parameters
    w_tvr: float = 60.0
    R_v: float = 0.09

    lcl_filter: LCLFilter9A = field(init=False)
    virtual_inertia: RotationalInertia2A = field(init=False)
    virtual_resistor: TransientVirtualResistor1A = field(init=False)

    def __post_init__(self):

        self.lcl_filter = LCLFilter9A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.virtual_inertia = RotationalInertia2A(self.h_s, self.kd_pu, self.wbase, alpha=self.alpha)
        self.virtual_resistor = TransientVirtualResistor1A(w_tvr=self.w_tvr, R_v=self.R_v)

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
            v_bus_mag=self.power_flow_variables.vmag_bus,
            relative_phase_deg=self.power_flow_variables.vphase_bus,
            p_bus=self.power_flow_variables.p_bus,
            q_bus=self.power_flow_variables.q_bus,
            reference_node="shunt",
        )

        self.virtual_inertia.get_steady_state(
            angle=lcl_init.angle_ref,
            w=1,
            p_ref=(lcl_init.v_sh_d * lcl_init.i_bus_d + lcl_init.v_sh_q * lcl_init.i_bus_q),
        )
        
        self.virtual_resistor.get_steady_state(
            i_d=lcl_init.i_bus_d,
            i_q=lcl_init.i_bus_q,
        )

    def define_variables_emt(self):
        # States 
        x = DynamicalVariables(
            name=["angle", "w", "z_tvr_d", "z_tvr_q", "i_vsc_a", "i_vsc_b", "i_vsc_c", "v_sh_a", "v_sh_b", "v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[  self.virtual_inertia.emt_init.angle,
                    self.virtual_inertia.emt_init.w,
                    self.virtual_resistor.emt_init.z_tvr_d,
                    self.virtual_resistor.emt_init.z_tvr_q,
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
            name=["p_ref", "v_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "grid", "grid", "grid"],
            init=[  self.virtual_inertia.emt_init.p_ref,
                    self.virtual_resistor.emt_init.v_ref,
                    self.lcl_filter.emt_init.v_bus_a,
                    self.lcl_filter.emt_init.v_bus_b,
                    self.lcl_filter.emt_init.v_bus_c]
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[  self.lcl_filter.emt_init.i_bus_a,
                    self.lcl_filter.emt_init.i_bus_b,
                    self.lcl_filter.emt_init.i_bus_c]
        )

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)
    
    def get_derivative_state_emt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Extract states
        angle, w, \
        z_tvr_d, z_tvr_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = x

        # Get inputs
        p_ref, v_ref, v_bus_a, v_bus_b, v_bus_c = u

        # Transform currents and voltages to dq reference frame
        i_vsc_d, i_vsc_q, _ = abc2dq0(i_vsc_a, i_vsc_b, i_vsc_c, angle)
        v_sh_d, v_sh_q, _ = abc2dq0(v_sh_a, v_sh_b, v_sh_c, angle)
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, angle)

        # Compute power at the shunt of the LCL filter
        p_sh = v_sh_d * i_bus_d + v_sh_q * i_bus_q
        q_sh = v_sh_q * i_bus_d - v_sh_d * i_bus_q

        # Compute voltage reference for the LCL filter
        v_vsc_d, v_vsc_q = self.virtual_resistor.get_algebraics_step_emt_dq0(v_ref=v_ref, i_d=i_bus_d, i_q=i_bus_q, z_tvr_d=z_tvr_d, z_tvr_q=z_tvr_q)

        # Transform voltage reference to abc reference frame
        v_vsc_a, v_vsc_b, v_vsc_c = dq02abc(v_vsc_d, v_vsc_q, 0, angle)

        # Compute derivatives of the state variables
        d_vi = self.virtual_inertia.get_derivatives_step_emt_abc(w, p_ref, p_sh)
        d_tvr = self.virtual_resistor.get_derivatives_step_emt_dq0(i_d=i_bus_d, i_q=i_bus_q, z_tvr_d=z_tvr_d, z_tvr_q=z_tvr_q)
        d_lcl= self.lcl_filter.get_derivatives_step_emt_abc(    i_vsc_a, i_vsc_b, i_vsc_c, 
                                                                v_sh_a, v_sh_b, v_sh_c, 
                                                                i_bus_a, i_bus_b, i_bus_c, 
                                                                v_vsc_a, v_vsc_b, v_vsc_c,
                                                                v_bus_a, v_bus_b, v_bus_c)

        return (d_vi + d_tvr + d_lcl)

    def plot_results_emt(self):

        angle, w, \
        z_tvr_d, z_tvr_q, \
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
            name=["angle", "w", "z_tvr_d", "z_tvr_q", "i_vsc_d", "i_vsc_q", 
                  "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q", "p_sh", "q_sh"],
            component=f"{self.type_}_{self.id}",
            value=[angle, w, z_tvr_d, z_tvr_q, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q,
                    p_sh, q_sh],
            time=tps
        )
        return results
    
    def get_output_emt(self, x: np.ndarray) -> np.ndarray:
        
        angle, w, \
        z_tvr_d, z_tvr_q, \
        i_vsc_a, i_vsc_b, i_vsc_c, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = x   

        return [i_bus_a, i_bus_b, i_bus_c]

    def get_interconnections_ssm(self, v_bus_D, v_bus_Q, i_bus_d, i_bus_q, relative_phase_rad):
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
        u_stack = [u_virtual_inertia, u_virtual_resistor, u_lcl_filter]
        y_stack = [y_virtual_inertia, y_virtual_resistor, y_lcl_filter]
        y_sys   = [Δi_bus_D, Δi_bus_Q]
        u_sys   = [Δp_ref, Δv_ref, Δv_bus_D, Δv_bus_Q]

        note that:
        u_virtual_inertia = [Δp_ref, Δi_bus_dq, Δv_sh_dq] (5 inputs)
        u_virtual_resistor = [Δi_bus_dq, Δv_ref] (3 inputs)
        u_lcl_filter = [Δv_vsc_dq, Δv_bus_dq, Δω] (5 inputs)
        
        y_virtual_inertia = [Δϕ, Δω] (2 outputs)
        y_virtual_resistor = [Δv_tvr_dq] (2 outputs)
        y_lcl_filter = [Δi_vsc_dq, Δi_bus_dq, Δv_sh_dq] (6 outputs)

        thus: u_stack has 5 + 3 + 5 = 13 inputs, y_stack has 2 + 2 + 6 = 10 outputs, y_sys has 2 outputs, and u_sys has 4 inputs.
        """

        angle = relative_phase_rad
        R = R_dq2DQ(angle)
        I = np.eye(2)

        a = d_DQ2dq_dangle(v_bus_D, v_bus_Q, angle).reshape(2,1)
        b = d_dq2DQ_dangle(i_bus_d, i_bus_q, angle).reshape(2,1)

        F = np.zeros((13, 10))
        G = np.zeros((13, 4))
        H = np.zeros((2, 10))
        L = np.zeros((2, 4))

        """
        Interconnection matrices
        Recall that to linearize the transformation from DQ to dq (and vice versa)
            Δv_dq = Uᵀ*(v_DQ)ₒ*Δϕ + Rᵀ*Δv_DQ
            Δi_DQ = U *(i_dq)ₒ*Δϕ + R *Δi_dq
        where
            R = [ cosϕₒ  -sinϕₒ ]
                [ sinϕₒ   cosϕₒ ]
            U = d/dϕₒ R
        and we define
            a := Uᵀ*(v_DQ)ₒ
            b := U *(i_dq)ₒ


        ┌ component ──▶             │ APC         ┆ TVR         ┆ LCL                                   │ Grid inputs
        │       ┌ index ──▶         │  0   1      ┆ 2,3         ┆ 4,5          6,7         8,9          │ 0         1       2,3
        │       │                   │  Δϕ  Δω     ┆ Δv_tvr_dq   ┆ Δi_vsc_dq    Δi_bus_dq   Δv_sh_dq     │ Δp_ref    Δv_ref  Δv_bus_DQ
        ▼       ▼                   │             │             │                                       │
        ────────────────────────────┼─────────────┴─────────────┴───────────────────────────────────────┼───────────────────────────────
        APC     0       Δp_ref      │  0   0        0               0              0           0        │   1         0       0
                1,2     Δi_bus_dq   │  0   0        0               0              I₂          0        │   0         0       0
                3,4     Δv_sh_dq    │  0   0        0               0              0           I₂       │   0         0       0
        TVR     5,6     Δi_bus_dq   │  0   0        0               0              I₂          0        │   0         0       0
                7       Δv_ref      │  0   0        0               0              0           0        │   0         1       0
        LCL     8,9     Δv_vsc_dq   │  0   0        I₂              0              0           0        │   0         0       0
                10,11   Δv_bus_dq   │  a   0        0               0              0           0        │   0         0       Rᵀ
                12      Δω          │  0   1        0               0              0           0        │   0         0       0
        ────────────────────────────┼───────────────────────────────────────────────────────────────────┼───────────────────────────────
        Grid    0,1     Δi_bus_DQ   │  b   0        0               0              R           0        │   0         0       0
        outputs
        """

        idx_F = [
            ([1, 2], [6, 7], I), ([3, 4], [8, 9], I), ([5, 6], [6, 7], I), ([8 ,9], [2, 3], I), 
            ([10, 11], [0], a), ([12], [1], 1),
        ]

        for rows, cols, value in idx_F:
            F[np.ix_(rows, cols)] = value

        idx_G = [
            ([0], [0], 1), ([7], [1], 1), ([10, 11], [2, 3], R.T),
        ]

        for rows, cols, value in idx_G:
            G[np.ix_(rows, cols)] = value

        H[:, [0]] = b
        H[np.ix_([0,1], [6,7])] = R

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

        virtual_resistor_ssm = self.virtual_resistor.get_small_signal_model(
            i_d = self.lcl_filter.emt_init.i_bus_d,
            i_q = self.lcl_filter.emt_init.i_bus_q,
            v_ref = self.virtual_resistor.emt_init.v_ref
        )

        lcl_filter_ssm = self.lcl_filter.get_small_signal_model(
            i_vsc_d = self.lcl_filter.emt_init.i_vsc_d,
            i_vsc_q = self.lcl_filter.emt_init.i_vsc_q,
            i_bus_d = self.lcl_filter.emt_init.i_bus_d,
            i_bus_q = self.lcl_filter.emt_init.i_bus_q,
            v_sh_d = self.lcl_filter.emt_init.v_sh_d,
            v_sh_q = self.lcl_filter.emt_init.v_sh_q
        )

        u = DynamicalVariables(
            name=["p_ref", "v_ref", "v_bus_D", "v_bus_Q"],
            type=["device", "device", "grid", "grid"],
            init=[self.virtual_inertia.emt_init.p_ref,
                  self.virtual_resistor.emt_init.v_ref,
                  self.lcl_filter.emt_init.v_bus_D,
                  self.lcl_filter.emt_init.v_bus_Q],
        )

        y = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            init=[self.lcl_filter.emt_init.i_bus_D,
                  self.lcl_filter.emt_init.i_bus_Q],
        )

        components = [virtual_inertia_ssm, virtual_resistor_ssm, lcl_filter_ssm]
        connections = self.get_interconnections_ssm(self.lcl_filter.emt_init.v_bus_D, 
                                                    self.lcl_filter.emt_init.v_bus_Q,
                                                    self.lcl_filter.emt_init.i_bus_d, 
                                                    self.lcl_filter.emt_init.i_bus_q,
                                                    self.lcl_filter.emt_init.angle_ref)
        self.ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        return self.ssm

    
    