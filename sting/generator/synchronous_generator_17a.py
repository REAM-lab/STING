import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import plotly.graph_objects as go
import polars as pl

from sting.components import (
    ExcitationSystem4A,
    ParallelRCShunt2A,
    SeriesRLBranch2A,
    SynchronousMachine7A,
    VoltageTransducer1A,
)
from sting.generator.core import Generator
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel
from sting.utils.transformations import abc2dq0, dq02abc, d_DQ2dq_dangle, R_DQ2dq, R_dq2DQ, d_dq2DQ_dangle


class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables


@dataclass(slots=True, kw_only=True, eq=False)
class SynchronousGenerator17A(Generator):
    """
    pass
    """
    # Machine circuits (in standard parameterization)
    x_d_pu: float 
    x_q_pu: float 
    x_td_pu: float
    x_tq_pu: float
    x_std_pu: float
    x_stq_pu: float
    t_td0_s: float
    t_tq0_s: float
    t_std0_s: float
    t_stq0_s: float
    x_l_pu: float
    r_a_pu: float
    x_0_pu: float
    # Transducer + exciter
    tau_v_s: float # Voltage filter time constant
    tb_s: float    # Lead lag compensator
    tc_s: float
    ka_pu: float   # Amplifier
    ta_s: float
    te_s: float    # Exciter
    ke_pu: float
    tf_s: float    # Feedback stabilizer
    kf_pu: float
    # RC shunt
    rsh_pu: float
    csh_pu: float
    # Transformer and branch to grid
    txr_power_MVA: float
    txr_voltage1_kV: float
    txr_voltage2_kV: float
    txr_r1_pu: float
    txr_x1_pu: float
    txr_r2_pu: float
    txr_x2_pu: float

    # Components
    machine: SynchronousMachine7A = field(init=False)
    transducer: VoltageTransducer1A = field(init=False)
    exciter: ExcitationSystem4A = field(init=False)
    rc_shunt: ParallelRCShunt2A = field(init=False)
    rl_branch: SeriesRLBranch2A = field(init=False)


    @property
    def rbr_pu(self):
        return (self.txr_r1_pu + self.txr_r2_pu) * self.base_power_MVA / self.txr_power_MVA

    @property
    def xbr_pu(self):
        return (self.txr_x1_pu + self.txr_x2_pu) * self.base_power_MVA / self.txr_power_MVA
    
    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz

    def __post_init__(self):
        self.machine = SynchronousMachine7A(
            x_d_pu=self.x_d_pu, x_q_pu=self.x_q_pu, x_l_pu=self.x_l_pu, r_a_pu=self.r_a_pu, 
            x_td_pu=self.x_td_pu, x_tq_pu=self.x_tq_pu, x_std_pu=self.x_std_pu, x_stq_pu=self.x_stq_pu,
            t_td0_s=self.t_td0_s, t_tq0_s=self.t_tq0_s, t_std0_s=self.t_std0_s, t_stq0_s=self.t_stq0_s,
            x_0_pu=self.x_0_pu, w_base=self.wbase
        )
        self.transducer = VoltageTransducer1A(tau_s=self.tau_v_s)
        self.exciter = ExcitationSystem4A(
            tb_s=self.tb_s, tc_s=self.tc_s, ka_pu=self.ka_pu, ta_s=self.ta_s, 
            te_s=self.te_s, ke_pu=self.ke_pu, tf_s=self.tf_s, kf_pu=self.kf_pu)
        self.rc_shunt = ParallelRCShunt2A(g_pu=1/self.rsh_pu, b_pu=self.csh_pu, wbase=self.wbase)
        self.rl_branch = SeriesRLBranch2A(r_pu=self.rbr_pu, x_pu=self.xbr_pu, wbase=self.wbase)

    def _calculate_emt_initial_conditions(self):
        # Unpack AC OPF solution
        v_bus_mag = self.power_flow_variables.vmag_bus
        relative_phase_deg = self.power_flow_variables.vphase_bus
        p_bus = self.power_flow_variables.p_bus
        q_bus = self.power_flow_variables.q_bus

        # Voltage in the end of the LCL filter
        phase_rad = relative_phase_deg * np.pi / 180
        v_bus_DQ = v_bus_mag * np.exp(phase_rad * 1j)
        # Current sent from the grid
        i_bus_DQ = (p_bus - q_bus * 1j) / np.conjugate(v_bus_DQ)
        # Voltage across the shunt element
        v_sh_DQ = v_bus_DQ + (self.rbr_pu + self.xbr_pu * 1j) * i_bus_DQ
        # Current flowing through shunt element
        i_sh_DQ = v_sh_DQ * (self.csh_pu * 1j) + v_sh_DQ / self.rsh_pu
        # Current sent from the synchronous machine
        i_sm_DQ = i_bus_DQ + i_sh_DQ

        v_ref_mag = np.abs(v_sh_DQ)
        p_ref = (i_sm_DQ*np.conj(v_sh_DQ)).real
        q_ref = -(i_sm_DQ*np.conj(v_sh_DQ)).imag

        # Compute all initial conditions
        self.rl_branch.get_steady_state(
            v_from_D=v_bus_DQ.real, v_from_Q=v_bus_DQ.imag,
            v_to_D=v_sh_DQ.real, v_to_Q=v_sh_DQ.imag,
            i_D=i_bus_DQ.real, i_Q=i_bus_DQ.imag,
        )
        self.rc_shunt.get_steady_state(
            i_D=i_sh_DQ.real, i_Q=i_sh_DQ.imag,
            v_D=v_sh_DQ.real, v_Q=v_sh_DQ.imag,
        )

        self.machine.get_steady_state(
            v_angle_deg = np.angle(v_sh_DQ, deg=True), 
            v_mag = v_ref_mag,
            p = p_ref,
            q = q_ref,
        )
        self.transducer.get_steady_state(v_d=v_sh_DQ.real, v_q=v_sh_DQ.imag)
        self.exciter.get_steady_state(v_ref=v_ref_mag, v_c=v_ref_mag, v_s=0)


    def _build_small_signal_model(self):
        pass

    def get_interconnections_ssm(self, i_stator_d, i_stator_q, v_shunt_D, v_shunt_Q, angle_rad):
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

            ['v_ref', 'v_c', 'v_s']

        ┌ component ──▶            | Machine                     ┆ Transducer ┆ Exciter                             ┆ Shunt     ┆ Branch     │ Grid inputs
        │       ┌ index ──▶        │ 0      1     2      3,4,5   ┆ 6      7      8,9    ┆ 10,11     ┆ 12,13      │ 0      1      2,3 
        ▼       ▼                  │ Δi_dq  Δi_0  Δi_fd  Δi_dq12 ┆ Δϕ  Δω ┆ Δp_m ┆ Δi_d  Δi_q  Δi_0  Δi_fd  Δi_1d  Δi_12q ┆ Δv_sh_DQ  ┆ Δi_bus_DQ  │ Δp_ref Δv_ref Δv_bus_DQ
        ───────────────────────────┼────────┴──────┴────────────────────────────────────────┴───────────┴────────────┼───────────────────────
        Shaft   0        Δp_m      │ 0   0    1      0     0     0     0      0      0         0           0         │ 0      0      0     
                1        Δi_d      │ 0   0    0      1     0     0     0      0      0         0           0         │ 0      0      0 
                2        Δi_q      │ 0   0    0      0     1     0     0      0      0         0           0         │ 0      0      0 
                3       -Δλ_q      │ 0   0    0      0     x_q   0     0      0     -x_aq      0           0         │ 0      0      0 
                4        Δλ_d      │ 0   0    0     -x_d   0     0     x_ad   x_ad   0         0           0         │ 0      0      0 
        Gov.    5        Δp_ref    │ 0   0    0      0     0     0     0      0      0         0           0         │ 1      0      0 
                6        Δω        │ 0   1    0      0     0     0     0      0      0         0           0         │ 0      0      0 
        Mach.   7,8      Δv_sh_dq  │ a   0    0      0     0     0     0      0      0         Rᵀ          0         │ 0      0      0 
                9        Δv_0      │ 0   0    0      0     0     0     0      0      0         0           0         │ 0      0      0 
                10       Δv_fd     │ 0   0    0      0     0     0     0      0      0         0           0         │ 0      1      0 
                11       Δω        │ 0   1    0      0     0     0     0      0      0         0           0         │ 0      0      0 
        Shunt   12,13    Δi_sh_DQ  │ b   0    0         R        0     0      0      0         0          -I₂        │ 0      0      0 
        Branch  14,15    Δv_sh_DQ  │ 0   0    0      0     0     0     0      0      0         I₂          0         │ 0      0      0 
                16,17    Δv_bus_DQ │ 0   0    0      0     0     0     0      0      0         0           0         │ 0      0      I₂
        ───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┼───────────────────────
        Grid    0,1      i_bus_DQ  │ 0   0    0      0     0     0     0      0      0         0           I₂        │ 0      0      0 
        outputs 
        """
        # Number of stacked/grid side inputs and outputs
        u_stack = 18
        y_stack = 14
        u_grid = 4
        y_grid = 2

        # Variables in the interconnections
        I = np.eye(2)
        a = d_DQ2dq_dangle(v_shunt_D, v_shunt_Q, angle_rad).reshape(2,1)
        b = d_dq2DQ_dangle(i_stator_d, i_stator_q, angle_rad).reshape(2,1)
        R = R_dq2DQ(angle_rad)
        x_d = self.machine.x_d_pu
        x_ad= self.machine.x_ad_pu
        x_q = self.machine.x_q_pu
        x_aq= self.machine.x_aq_pu

        # Interconnection matrices
        L11 = np.zeros((u_stack, y_stack))
        L12 = np.zeros((u_stack, u_grid))
        L21 = np.zeros((y_grid, y_stack))
        L22 = np.zeros((y_grid, u_grid))

        # Row, column, value tuples for each matrix
        idx_11 = [
            ([0],[2],1), ([1,2],[3,4],I), ([3],[4],x_q), ([3],[9],-x_aq), ([3],[8],-x_aq), ([4],[3],-x_d), 
            ([4],[6],x_ad), ([4],[7],x_ad), ([6],[1],1), ([7,8],[0],a), ([7,8],[10,11],R.T), ([11], [1], 1), 
            ([12,13],[0],b), ([12,13],[3,4],R), ([12,13],[12,13],-I), ([14,15],[10,11],I)]

        idx_12 = [([5],[0],1), ([10],[1],1),([16,17],[2,3],I)]
        idx_21 = [([0,1],[12,13],I)]

        # Fill out each matrix
        matrix_index_pairs =  [(L11, idx_11), (L12, idx_12), (L21, idx_21)]
        for matrix, idx in matrix_index_pairs:
            for rows, cols, value in idx:
                matrix[np.ix_(rows, cols)] = value

        return (L11,L12,L21,L22)

    def _build_quadratic_bilinear_model(self):
        pass

    def define_variables_emt(self):
        states = [
            ("angle", self.machine.emt_init.angle),
            # Synchronous machine
            ("i_stator_d", self.machine.emt_init.i_d),
            ("i_stator_q", self.machine.emt_init.i_q),
            ("i_stator_0", self.machine.emt_init.i_0),
            ("i_field_d", self.machine.emt_init.i_fd),
            ("i_damper_1d", self.machine.emt_init.i_1d),  
            ("i_damper_1q", self.machine.emt_init.i_1q),  
            ("i_damper_2q", self.machine.emt_init.i_2q),  
            # Transducer + exciter
            ("transducer_vmag", self.transducer.emt_init.v_c1),
            ("exciter_leadlag", self.exciter.emt_init.x_l),
            ("exciter_amplifier", self.exciter.emt_init.x_a),
            ("exciter_exciter", self.exciter.emt_init.x_e),
            ("exciter_damper", self.exciter.emt_init.x_f),
            # RC shunt
            ("v_shunt_a", self.rc_shunt.emt_init.v_a),
            ("v_shunt_b", self.rc_shunt.emt_init.v_b),
            ("v_shunt_c", self.rc_shunt.emt_init.v_c),
            # RL branch
            ("i_bus_a", self.rl_branch.emt_init.i_a),
            ("i_bus_b", self.rl_branch.emt_init.i_b),
            ("i_bus_c", self.rl_branch.emt_init.i_c),
        ]
        # States 
        name, init = map(list, zip(*states))
        x = DynamicalVariables(name=name, init=init, component=f"{self.type_}_{self.id}")


        init = self.rl_branch.emt_init

        # Inputs 
        u = DynamicalVariables(
            name=["v_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "grid", "grid", "grid"],
            init=[
                self.exciter.emt_init.v_ref, 
                init.v_to_a, init.v_to_b, init.v_to_c]
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[init.i_a, init.i_b, init.i_c]
        )
        
        self.variables_emt = VariablesEMT(x=x,u=u,y=y)
    
    def get_derivative_state_emt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Unpacking states and inputs
        angle, \
        i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, \
        v_c, x_l, x_a, x_e, x_f, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = x

        v_ref, v_bus_a, v_bus_b, v_bus_c = u

        # -------------------- #
        # Algebraic operations #
        # -------------------- #
        # Generator rotates at w_base
        w = 1
        # Compute the exciter output field voltage by adding the initial field voltage back
        v_fd = x_e + self.machine.emt_init.v_fd
        # Shunt voltage abc to dq0
        v_sh_d, v_sh_q, v_sh_0 = abc2dq0(v_sh_a, v_sh_b, v_sh_c, angle)
        # Synchronous machine dq0 to abc
        i_sm_a, i_sm_b, i_sm_c = dq02abc(i_d, i_q, i_0, angle)
        # Flow of current into the shunt by KCL
        i_sh_a = i_sm_a - i_bus_a
        i_sh_b = i_sm_b - i_bus_b
        i_sh_c = i_sm_c - i_bus_c
        
        # ----------------------- #
        # Differential operations #
        # ----------------------- #
        dx = [self.wbase]
        dx += self.machine.get_derivatives_step_emt_dq0(
            i_d=i_d, i_q=i_q, i_0=i_0, i_fd=i_fd, i_1d=i_1d, i_1q=i_1q, i_2q=i_2q,
            v_d=v_sh_d, v_q=v_sh_q, v_0=v_sh_0, v_fd=v_fd, w=w
            )
        dx += self.transducer.get_derivatives_step_emt_dq0(v_c1=v_c, v_d=v_sh_d, v_q=v_sh_q)
        dx += self.exciter.get_derivatives_step_emt_dq0(
            x_l=x_l, x_a=x_a, x_e=x_e, x_f=x_f, 
            v_ref=v_ref, v_c=v_c, v_s=0
            )
        dx += self.rc_shunt.get_derivatives_step_emt_abc(
            v_sh_a=v_sh_a, v_sh_b=v_sh_b, v_sh_c=v_sh_c, 
            i_sh_a=i_sh_a, i_sh_b=i_sh_b, i_sh_c=i_sh_c
            )
        dx += self.rl_branch.get_derivatives_step_emt_abc(
            i_a=i_bus_a, i_b=i_bus_b, i_c=i_bus_c,
            v_from_a=v_sh_a, v_from_b=v_sh_b, v_from_c=v_sh_c, 
            v_to_a=v_bus_a, v_to_b=v_bus_b, v_to_c=v_bus_c
            )

        return dx


    def get_output_emt(self, x: np.ndarray) -> np.ndarray:
        angle, \
        i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, \
        v_c, x_l, x_a, x_e, x_f, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = x

        return [i_bus_a, i_bus_b, i_bus_c]
    
    def plot_results_emt(self):
        angle, \
        i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, \
        v_c, x_l, x_a, x_e, x_f, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value

        # Transform abc to dq0
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, angle)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, angle)])

        names = [
            "angle",
            "i_stator_d", "i_stator_q", "i_stator_0", "i_field_d", "i_damper_1d", "i_damper_1q", "i_damper_2q", 
            "transducer_vmag","exciter_leadlag","exciter_amplifier","exciter_exciter","exciter_damper", 
            "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q"
        ]
        values = [
            angle,
            i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q,
            v_c, x_l, x_a, x_e, x_f, 
            v_sh_d, v_sh_q, i_bus_d, i_bus_q 
            ]

        results = DynamicalVariables(
            name=names,
            component=f"{self.type_}_{self.id}",
            value=values,
            time=self.variables_emt.x.time
        )
        return results
