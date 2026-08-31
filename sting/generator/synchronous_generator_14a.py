from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.components import (
    ParallelRCShunt2A,
    RotationalInertia2A,
    SeriesRLBranch2A,
    SpeedGovernor1A,
    SynchronousMachine7A,
)
from sting.generator.core import Generator
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel, QuadraticBilinearModel
from sting.utils.transformations import dq02abc, abc2dq0, R_dq2DQ, R_DQ2dq, d_dq2DQ_dangle, d_DQ2dq_dangle

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables


@dataclass(slots=True, kw_only=True, eq=False)
class SynchronousGenerator14A(Generator):
    """
    Models a synchronous generator consisting of a:
    - 7th order machine
    - 3rd order governor + rotational inertia
    - 4th order RC shunt + transformer
    """
    # Shaft
    h_s: float
    kd_w_pu: float
    alpha: float = 0
    # Governor
    tau_g_s: float  # Time constant (in seconds)
    kr_pu: float    # Speed regulator gain (in pu)
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
    shaft: RotationalInertia2A = field(init=False)
    governor: SpeedGovernor1A = field(init=False)
    machine: SynchronousMachine7A = field(init=False)
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
        self.shaft = RotationalInertia2A(h_s=self.h_s, kd_w_pu=self.kd_w_pu, w_nom=self.wbase, alpha=self.alpha)
        self.governor = SpeedGovernor1A(tau_s=self.tau_g_s, kr_pu=self.kr_pu)
        self.machine = SynchronousMachine7A(
            x_d_pu=self.x_d_pu, x_q_pu=self.x_q_pu, x_l_pu=self.x_l_pu, r_a_pu=self.r_a_pu, 
            x_td_pu=self.x_td_pu, x_tq_pu=self.x_tq_pu, x_std_pu=self.x_std_pu, x_stq_pu=self.x_stq_pu,
            t_td0_s=self.t_td0_s, t_tq0_s=self.t_tq0_s, t_std0_s=self.t_std0_s, t_stq0_s=self.t_stq0_s,
            x_0_pu=self.x_0_pu, w_base=self.wbase
        )
        self.rc_shunt = ParallelRCShunt2A(g_pu=1/self.rsh_pu, b_pu=self.csh_pu, wbase=self.wbase)
        self.rl_branch = SeriesRLBranch2A(r_pu=self.rbr_pu, x_pu=self.xbr_pu, wbase=self.wbase)
        self.phase_angle_name = self.shaft.phase_angle_name

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
        sm_init = self.machine.get_steady_state(
            v_angle_deg = np.angle(v_sh_DQ, deg=True), 
            v_mag = v_ref_mag,
            p = p_ref,
            q = q_ref,
        )
        # Initial per unit mechanical torque by balancing torques
        t_e = self.machine.electrical_torque(
            i_d=sm_init.i_d, i_fd=sm_init.i_fd, i_1d=sm_init.i_1d,
            i_q=sm_init.i_q, i_1q=sm_init.i_1q, i_2q=sm_init.i_2q)
        
        # Governor states are the change relative to nominal
        self.governor.get_steady_state(p_ref=0, w=0)
        self.shaft.get_steady_state(p_ref=t_e, angle=self.machine.emt_init.angle, w=1)


    def _build_small_signal_model(self):

        # Initial conditions of the machine
        i_d = self.machine.emt_init.i_d
        i_q = self.machine.emt_init.i_q
        i_fd = self.machine.emt_init.i_fd
        i_1d = self.machine.emt_init.i_1d
        i_1q = self.machine.emt_init.i_1q
        i_2q = self.machine.emt_init.i_2q
        angle = self.machine.emt_init.angle

        psi_d = -self.machine.x_d_pu*i_d + self.machine.x_ad_pu*(i_fd + i_1d)
        psi_q = -self.machine.x_q_pu*i_q + self.machine.x_aq_pu*(i_1q + i_2q)
        t_e = self.machine.electrical_torque(
            i_d=i_d, i_fd=i_fd, i_1d=i_1d, i_q=i_q, i_1q=i_1q, i_2q=i_2q
        )
        # T_e = λ_d*i_q - λ_q*i_d
        shaft_ssm = self.shaft.get_small_signal_model(
            i_d=i_d, i_q=i_q, v_d=-psi_q, v_q=psi_d, p_ref=t_e, angle=angle
        )
        # Note: Governor states are the change relative to nominal
        governor_ssm = self.governor.get_small_signal_model(
            x_gov=self.governor.emt_init.x_gov, p_ref=0, w=0
        )
        machine_ssm = self.machine.get_small_signal_model(
            i_d=i_d, i_q=i_q, i_0=0, i_fd=i_fd, i_1d=i_1d, i_1q=i_1q, i_2q=i_2q,
            v_d=self.machine.emt_init.v_d, v_q=self.machine.emt_init.v_q, v_0=0, 
            v_fd=self.machine.emt_init.v_fd, w=1
        )
        shunt_ssm = self.rc_shunt.get_small_signal_model(
            v_D=self.rc_shunt.emt_init.v_D,
            v_Q=self.rc_shunt.emt_init.v_Q, 
            i_D=self.rc_shunt.emt_init.i_D,
            i_Q=self.rc_shunt.emt_init.i_Q,  
        )
        branch_ssm = self.rl_branch.get_small_signal_model(
            v_from_D=self.rl_branch.emt_init.v_from_D,
            v_from_Q=self.rl_branch.emt_init.v_from_Q,
            v_to_D=self.rl_branch.emt_init.v_to_D,
            v_to_Q=self.rl_branch.emt_init.v_to_Q,
            i_D=self.rl_branch.emt_init.i_D,
            i_Q=self.rl_branch.emt_init.i_Q,
        )

        u = DynamicalVariables(
            name=["p_ref", "v_ref", "v_bus_D", "v_bus_Q"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "grid", "grid"],
            init=[
                self.shaft.emt_init.p_ref, 
                self.machine.emt_init.v_fd, 
                self.rl_branch.emt_init.v_to_D,
                self.rl_branch.emt_init.v_to_Q]
        )

        y = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            component=f"{self.type_}_{self.id}",
            init=[self.rl_branch.emt_init.i_D, self.rl_branch.emt_init.i_Q]
        )

        # Generate small-signal model
        components = [shaft_ssm, governor_ssm, machine_ssm, shunt_ssm, branch_ssm]
        connections = self.get_interconnections_ssm(
            i_stator_d=i_d, 
            i_stator_q=i_q, 
            v_shunt_D=self.rc_shunt.emt_init.v_D, 
            v_shunt_Q=self.rc_shunt.emt_init.v_Q, 
            angle_rad=angle)
        self.ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        return self.ssm

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

        The air gap torque of the machine is
            T_e = λ_d*i_q - λ_q*i_d

            λ_d = -x_d*i_d + x_ad*(i_fd + i_1d)
           -λ_q = x_q*i_q - x_aq*(i_1q + i_2q)

        And by Kirchhoff's current law (KCL)
            i_sh = i_sm - i_bus

            

        ┌ component ──▶            | Shaft  ┆ Gov. ┆ Machine                                ┆ Shunt     ┆ Branch     │ Grid inputs
        │       ┌ index ──▶        │ 0   1  ┆  2   ┆ 3     4     5     6      7      8,9    ┆ 10,11     ┆ 12,13      │ 0      1      2,3 
        ▼       ▼                  │ Δϕ  Δω ┆ Δt_m ┆ Δi_d  Δi_q  Δi_0  Δi_fd  Δi_1d  Δi_12q ┆ Δv_sh_DQ  ┆ Δi_bus_DQ  │ Δp_ref Δv_ref Δv_bus_DQ
        ───────────────────────────┼────────┴──────┴────────────────────────────────────────┴───────────┴────────────┼───────────────────────
        Shaft   0        Δt_m      │ 0   0    1      0     0     0     0      0      0         0           0         │ 0      0      0     
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
         # Initial conditions of the machine
        i_d = self.machine.emt_init.i_d
        i_q = self.machine.emt_init.i_q
        i_fd = self.machine.emt_init.i_fd
        i_1d = self.machine.emt_init.i_1d
        i_1q = self.machine.emt_init.i_1q
        i_2q = self.machine.emt_init.i_2q
        angle = self.machine.emt_init.angle

        t_e = self.machine.electrical_torque(
            i_d=i_d, i_fd=i_fd, i_1d=i_1d, i_q=i_q, i_1q=i_1q, i_2q=i_2q
        )
        # T_e = λ_d*i_q - λ_q*i_d
        shaft_ssm = self.shaft.get_quadratic_bilinear_model(
            angle_rad=angle, w=1, p_ref=t_e, p=t_e
        )
        # Note: Governor states are the change relative to nominal
        governor_ssm = self.governor.get_quadratic_bilinear_model(
            x_gov=self.governor.emt_init.x_gov, p_ref=0, w=0
        )
        machine_ssm = self.machine.get_quadratic_bilinear_model(
            i_d=i_d, i_q=i_q, i_0=0, i_fd=i_fd, i_1d=i_1d, i_1q=i_1q, i_2q=i_2q,
            v_d=self.machine.emt_init.v_d, v_q=self.machine.emt_init.v_q, v_0=0, 
            v_fd=self.machine.emt_init.v_fd, w=1
        )
        shunt_ssm = self.rc_shunt.get_quadratic_bilinear_model(
            v_D=self.rc_shunt.emt_init.v_D,
            v_Q=self.rc_shunt.emt_init.v_Q, 
            i_D=self.rc_shunt.emt_init.i_D,
            i_Q=self.rc_shunt.emt_init.i_Q,  
        )
        branch_ssm = self.rl_branch.get_quadratic_bilinear_model(
            v_from_D=self.rl_branch.emt_init.v_from_D,
            v_from_Q=self.rl_branch.emt_init.v_from_Q,
            v_to_D=self.rl_branch.emt_init.v_to_D,
            v_to_Q=self.rl_branch.emt_init.v_to_Q,
            i_D=self.rl_branch.emt_init.i_D,
            i_Q=self.rl_branch.emt_init.i_Q,
        )

        u = DynamicalVariables(
            name=["p_ref", "v_ref", "one", "v_bus_D", "v_bus_Q"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "device", "grid", "grid"],
            init=[
                self.shaft.emt_init.p_ref, 
                self.machine.emt_init.v_fd, 
                1,
                self.rl_branch.emt_init.v_to_D,
                self.rl_branch.emt_init.v_to_Q]
        )

        y = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            component=f"{self.type_}_{self.id}",
            init=[self.rl_branch.emt_init.i_D, self.rl_branch.emt_init.i_Q]
        )

        # Generate small-signal model
        components = [shaft_ssm, governor_ssm, machine_ssm, shunt_ssm, branch_ssm]
        connections = self.get_interconnections_qbm(t_m=t_e, p_ref=t_e)
        self.qbm = QuadraticBilinearModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        return self.qbm


    def get_interconnections_qbm(self, t_m, p_ref):
        """       
        Interconnection matrices
        ------------------------
        By Kirchhoff's current law (KCL)
            i_sh = i_sm - i_bus

        ┌ component ──▶           | Shaft      ┆ Gov. ┆ Machine                               ┆ Shunt   ┆ Branch   │ Grid inputs
        │       ┌ index ──▶       │ 0  1   2   ┆  3   ┆ 4    5     6   7     8     9    10    ┆ 11,12   ┆ 13,14    │ 0      1      2    3,4 
        ▼       ▼                 │ ω  sin cos ┆ Δt_m ┆ i_d  i_q  i_0  i_fd  i_1d  i_1q  i_2q ┆ v_sh_DQ ┆ i_bus_DQ │ p_ref  v_ref  one  v_bus_DQ
        ──────────────────────────┼────────────┴──────┴───────────────────────────────────────┴─────────┴──────────┼───────────────────────
        Shaft   0        t_m      │ 0  0   0     1      0    0    0    0     0     0     0      0         0        │ 0      0    t_m(0) 0
                1        one      │ 0  0   0     0      0    0    0    0     0     0     0      0         0        │ 0      0      1    0
                2       *t_e      │ 0  0   0     0      0    0    0    0     0     0     0      0         0        │ 0      0      0    0
        Gov.    3        Δp_ref   │ 0  0   0     0      0    0    0    0     0     0     0      0         0        │ 1      0    -p(0)  0
                4        Δω       │ 1  0   0     0      0    0    0    0     0     0     0      0         0        │ 0      0     -1    0
        Mach.   5,6     *v_sh_dq  │ 0  0   0     0      0    0    0    0     0     0     0      0         0        │ 0      0      0    0
                7        v_0      │ 0  0   0     0      0    0    0    0     0     0     0      0         0        │ 0      0      0    0
                8        v_fd     │ 0  0   0     0      0    0    0    0     0     0     0      0         0        │ 0      1      0    0
                9        ω        │ 1  0   0     0      0    0    0    0     0     0     0      0         0        │ 0      0      0    0
        Shunt   10,11   *i_sh_DQ  │ 0  0   0     0      0    0    0    0     0     0     0      0        -I₂       │ 0      0      0    0
        Branch  12,13    v_sh_DQ  │ 0  0   0     0      0    0    0    0     0     0     0      I₂        0        │ 0      0      0    0
                14,15    v_bus_DQ │ 0  0   0     0      0    0    0    0     0     0     0      0         0        │ 0      0      0    I₂
        ──────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────────────
        Grid    0,1      i_bus_DQ │ 0   0    0      0     0     0     0      0      0         0           I₂       │ 0      0      0    0
        outputs 

        
        idx_11 = [([0],[3],1), ([4],[0],1),([9],[0],1), ([10,11],[13,14],-I), ([12,13],[11,12],I)]
        idx_12 = [([0],[2],t_m), ([1],[2],1), ([3],[0],1), ([3],[2],-p_ref), ([4],[2],-1), ([8],[1],1), ([14,15],[3,4],I)]
        idx_21 = [([0,1],[13,14],I)]
        

        Recall the transformation from DQ to dq  
            i_d =  i_D*cos + i_Q*sin
            i_q = -i_D*sin + i_Q*cos
        
        The air gap torque of the machine is
            T_e = λ_d*i_q - λ_q*i_d
        where
            λ_d = -x_d*i_d + x_ad*(i_fd + i_1d)
           -λ_q = x_q*i_q - x_aq*(i_1q + i_2q)

        We will define
            J = [ 0  1]
                [-1  0]
            
                                   | Shaft       ┆ Gov.┆ Machine            ┆ Shunt
                             1     │ 0   1   2   ┆ 3   ┆ 4,5  6   7,8,9,10  ┆ 11,12    
        (x_1 * x)            sin * │ ω   sin cos ┆ t_m ┆ i_dq i_0 i_dampers ┆ v_sh_DQ  ....
        ───────────────────────────┼─────────────┴─────┴────────────────────┴──────────
        Mach.   5,6     *v_sh_dq   │ 0   0   0     0     0    0   0           J₂             
        Shunt   10,11   *i_sh_DQ   │ 0   0   0     0    -J₂   0   0           0 
        
                                   | Shaft       ┆ Gov.┆ Machine            ┆ Shunt
                             2     │ 0   1   2   ┆ 3   ┆ 4,5  6   7,8,9,10  ┆ 11,12    
        (x_2 * x)            cos * │ ω   sin cos ┆ t_m ┆ i_dq i_0 i_dampers ┆ v_sh_DQ  ....
        ───────────────────────────┼─────────────┴─────┴────────────────────┴──────────
        Mach.   5,6     *v_sh_dq   │ 0   0   0     0     0    0   0           I₂                
        Shunt   10,11   *i_sh_DQ   │ 0   0   0     0     I₂   0   0           0 
        
        idx_x1 = [([5,6],[11,12],J),([10,11],[4,5],-J)]
        idx_x2 = [([5,6],[11,12],I),([10,11],[4,5],I)

                                   | Shaft+Gov ┆ Machine         
                             2     │ 0,1,2,3   ┆ 4    5     6   7     8     9    10  
        (x_4 * x)            i_d * │ ...       ┆ i_d  i_q  i_0  i_fd  i_1d  i_1q  i_2q 
        ───────────────────────────┼───────────┴─────────────────────────────────────
        Shaft   2           *t_e   │ 0           0    x_q  0    0     0    -x_aq  -x_aq
        
                                   | Shaft+Gov ┆ Machine         
                             2     │ 0,1,2,3   ┆ 4    5     6   7     8     9    10  
        (x_5 * x)            i_q * │ ...       ┆ i_d  i_q  i_0  i_fd  i_1d  i_1q  i_2q 
        ───────────────────────────┼───────────┴─────────────────────────────────────
        Shaft   2           *t_e   │ 0          -x_d  0    0    x_ad  x_ad  0     0   

        idx_x4 = [([2],[5],x_q), ([2],[9],-x_aq), ([2],[10],-x_aq)]
        idx_x5 = [([2],[4],-x_d), ([2],[7],x_ad), ([2],[8],x_ad)]
        """
        # Number of stacked/grid side inputs and outputs
        u_stack = 16
        y_stack = 15
        x_stack = 15
        u_grid = 5
        y_grid = 2
        # Variables in the interconnections
        I = np.eye(2)
        J = np.array([[0, 1], [-1,0]])
        x_d = self.machine.x_d_pu
        x_ad= self.machine.x_ad_pu
        x_q = self.machine.x_q_pu
        x_aq= self.machine.x_aq_pu

        # Linear interconnection matrices
        L11 = np.zeros((u_stack, y_stack))
        L12 = np.zeros((u_stack, u_grid))
        L21 = np.zeros((y_grid, y_stack))
        L22 = np.zeros((y_grid, u_grid))


        # Nonlinear interconnection matrices
        M1_x1, M1_x2, M1_x4, M1_x5 = (np.zeros((u_stack, x_stack)) for _ in range(4))
        M2 = np.zeros((u_stack, x_stack*u_grid))
        
        idx_11 = [([0],[3],1), ([4],[0],1),([9],[0],1), ([10,11],[13,14],-I), ([12,13],[11,12],I)]
        idx_12 = [([0],[2],t_m), ([1],[2],1), ([3],[0],1), ([3],[2],-p_ref), ([4],[2],-1), ([8],[1],1), ([14,15],[3,4],I)]
        idx_21 = [([0,1],[13,14],I)]

        idx_x1 = [([5,6],[11,12],J),([10,11],[4,5],-J)]
        idx_x2 = [([5,6],[11,12],I),([10,11],[4,5],I)]
        idx_x4 = [([2],[5],x_q), ([2],[9],-x_aq), ([2],[10],-x_aq)]
        idx_x5 = [([2],[4],-x_d), ([2],[7],x_ad), ([2],[8],x_ad)]

        # Fill out each matrix
        matrix_index_pairs =  [(L11, idx_11), (L12, idx_12), (L21, idx_21), (M1_x1, idx_x1), (M1_x2, idx_x2), (M1_x4, idx_x4), (M1_x5, idx_x5)]
        for matrix, idx in matrix_index_pairs:
            for rows, cols, value in idx:
                matrix[np.ix_(rows, cols)] = value

        # Stack matrices in M1
        M1 = np.hstack([np.zeros((u_stack, x_stack)), M1_x1, M1_x2, np.zeros((u_stack, x_stack)), M1_x4, M1_x5, np.zeros((u_stack, x_stack*9))])
                        
        return (L11, L12, L21, L22, M1, M2)


    def define_variables_emt(self):
        states = [
            # Rotational inertia of the shaft
            ("angle", self.shaft.emt_init.angle),
            ("w", self.shaft.emt_init.w),
            # Governor
            ("governor", self.governor.emt_init.x_gov),
            # Synchronous machine
            ("i_stator_d", self.machine.emt_init.i_d),
            ("i_stator_q", self.machine.emt_init.i_q),
            ("i_stator_0", self.machine.emt_init.i_0),
            ("i_field_d", self.machine.emt_init.i_fd),
            ("i_damper_1d", self.machine.emt_init.i_1d),  
            ("i_damper_1q", self.machine.emt_init.i_1q),  
            ("i_damper_2q", self.machine.emt_init.i_2q),  
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
            name=["p_ref", "v_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "grid", "grid", "grid"],
            init=[
                self.shaft.emt_init.p_ref, 
                self.machine.emt_init.v_fd, 
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
        angle, w, x_gov, \
        i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = x

        p_ref, v_ref, v_bus_a, v_bus_b, v_bus_c = u

        # -------------------- #
        # Algebraic operations #
        # -------------------- #
        delta_p_ref = p_ref - self.shaft.emt_init.p_ref
        delta_w = w - 1
        # Shunt voltage abc to dq0
        v_sh_d, v_sh_q, v_sh_0 = abc2dq0(v_sh_a, v_sh_b, v_sh_c, angle)
        # Synchronous machine dq0 to abc
        i_sm_a, i_sm_b, i_sm_c = dq02abc(i_d, i_q, i_0, angle)
        # Flow of current into the shunt by KCL
        i_sh_a = i_sm_a - i_bus_a
        i_sh_b = i_sm_b - i_bus_b
        i_sh_c = i_sm_c - i_bus_c
        # Mechanical torque
        t_m = self.shaft.emt_init.p_ref + x_gov
        # Air gap torque
        t_e = self.machine.electrical_torque( 
            i_d=i_d, i_q=i_q, i_fd=i_fd, i_1d=i_1d, i_1q=i_1q, i_2q=i_2q)
        # ----------------------- #
        # Differential operations #
        # ----------------------- #
        dx = []
        dx += self.shaft.get_derivatives_step_emt_abc(w=w, p_ref=t_m, p=t_e)
        dx += self.governor.get_derivatives_step_emt(x_gov=x_gov, p_ref=delta_p_ref, w=delta_w)
        dx += self.machine.get_derivatives_step_emt_dq0(
            i_d=i_d, i_q=i_q, i_0=i_0, i_fd=i_fd, i_1d=i_1d, i_1q=i_1q, i_2q=i_2q,
            v_d=v_sh_d, v_q=v_sh_q, v_0=v_sh_0, v_fd=v_ref, w=w
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
        angle, w, x_gov, \
        i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = x

        return [i_bus_a, i_bus_b, i_bus_c]
    
    def plot_results_emt(self):
        angle, w, x_gov, \
        i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, \
        v_sh_a, v_sh_b, v_sh_c, \
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value

        # Transform abc to dq0

        grid_angle = self.wbase*self.variables_emt.x.time
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, grid_angle)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, grid_angle)])

        names = [
            "angle", "w", 
            "governor",
            "i_stator_d", "i_stator_q", "i_stator_0", "i_field_d", "i_damper_1d", "i_damper_1q", "i_damper_2q", 
            "v_shunt_D", "v_shunt_Q", "i_bus_D", "i_bus_Q"
        ]
        values = [
            angle, w,
            x_gov, 
            i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q,
            v_sh_d, v_sh_q, i_bus_d, i_bus_q 
            ]

        results = DynamicalVariables(
            name=names,
            component=f"{self.type_}_{self.id}",
            value=values,
            time=self.variables_emt.x.time
        )
        return results
