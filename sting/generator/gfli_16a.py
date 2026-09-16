"""
This module implements a 16th order Grid-following Inverter (GFLI) comprised of:
- 3rd order PLL with filter: It that tracks the phase of the grid voltage.
- 1st order active power controller: A PI controller that regulates the active power of the inverter.
- 1st order reactive power controller: A PI controller that regulates the reactive power of the inverter.
- 2nd order current controller: A dq-based frame PI controller
- 9th order LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
"""
from dataclasses import dataclass, field

import numpy as np

from sting.branch.series_rl_branch_2a import SeriesRLBranch2A
from sting.components import (
    ActivePowerPI1A,
    InnerCurrentController2A,
    LCLFilter9A,
    PhaseLockedLoop3A,
    ReactivePowerPI1A,
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


@dataclass(slots=True, kw_only=True, eq=False)
class GFLI16A(Generator):
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
    kp_pll_rad_s: float
    ki_pll_rad2_s2: float
    tau_pll_s: float
    alpha: float = 0
    # Current controller parameters
    kp_cc_pu: float
    ki_cc_puHz: float
    kff_cc: float
    # Power controller parameters
    kp_pc_pu: float
    ki_pc_puHz: float

    # Components
    lcl_filter: LCLFilter9A = field(init=False)
    # LCL filter components for quadratic bilinear model
    lcl_br1: SeriesRLBranch2A = field(init=False)
    lcl_br2: SeriesRLBranch2A = field(init=False)
    lcl_sh: ParallelRCShunt2A  = field(init=False)
    current_controller: InnerCurrentController2A = field(init=False)
    phase_locked_loop: PhaseLockedLoop3A = field(init=False)
    active_power_controller: ActivePowerPI1A = field(init=False)
    reactive_power_controller: ReactivePowerPI1A = field(init=False)

    def __post_init__(self):
        self.lcl_filter = LCLFilter9A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.lcl_br1 = SeriesRLBranch2A(r_pu=self.rf1_pu, x_pu=self.xf1_pu, base_frequency_Hz=self.base_frequency_Hz)
        self.lcl_br2 = SeriesRLBranch2A(r_pu=self.rf2_pu, x_pu=self.xf2_pu, base_frequency_Hz=self.base_frequency_Hz)
        self.lcl_sh = ParallelRCShunt2A(g_pu=1/self.rsh_pu, b_pu=self.csh_pu, base_frequency_Hz=self.base_frequency_Hz)
        self.phase_locked_loop = PhaseLockedLoop3A(self.kp_pll_rad_s, self.ki_pll_rad2_s2, self.tau_pll_s, self.wbase, alpha=self.alpha)
        self.current_controller = InnerCurrentController2A(self.kp_cc_pu, self.ki_cc_puHz, self.kff_cc, self.xf1_pu + self.xf2_pu)
        self.active_power_controller = ActivePowerPI1A(kp_pu=self.kp_pc_pu, ki_puHz=self.ki_pc_puHz)
        self.reactive_power_controller = ReactivePowerPI1A(kp_pu=self.kp_pc_pu, ki_puHz=self.ki_pc_puHz)
        self.phase_angle_name = self.phase_locked_loop.phase_angle_name

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


        ┌ component ──▶           │ PLL    ┆ APC      ┆ RPC      ┆ ICC       ┆ LCL                            │ Grid inputs
        │       ┌ index ──▶       │ 0   1  ┆ 2        ┆ 3        ┆ 4,5       ┆ 6,7        8,9        10,11    │ 0       1       2,3
        ▼       ▼                 │ Δω  Δϕ ┆ Δi_ref_d ┆ Δi_ref_q ┆ Δv_vsc_dq ┆ Δi_vsc_dq  Δi_bus_dq  Δv_sh_dq │ Δp_ref  Δq_ref  Δv_bus_DQ
        ──────────────────────────┼────────┴──────────┴──────────┴───────────┴────────────────────────────────┼────────────────────────────
        PLL     0,1     Δv_bus_DQ │  0  0    0          0          0           0          0          0        │ 0       0       I₂
        APC     2       Δp_ref    │  0  0    0          0          0           0          0          0        │ 1       0       0
                3,4     Δi_bus_dq │  0  0    0          0          0           0          I₂         0        │ 0       0       0
                5,6     Δv_bus_dq │  0  a    0          0          0           0          0          0        │ 0       0       Rᵀ
        RPC     7       Δq_ref    │  0  0    0          0          0           0          0          0        │ 0       1       0
                8,9     Δi_bus_dq │  0  0    0          0          0           0          I₂         0        │ 0       0       0
                10,11   Δv_bus_dq │  0  a    0          0          0           0          0          0        │ 0       0       Rᵀ
        IC      12      Δi_ref_d  │  0  0    1          0          0           0          0          0        │ 0       0       0
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

    def _build_quadratic_bilinear_model(self):
        # Unpack OPF solutions
        v_mag, phase_deg = self.power_flow_variables.vmag_bus, self.power_flow_variables.vphase_bus
        p_bus, q_bus = self.power_flow_variables.p_bus, self.power_flow_variables.q_bus
        # Initial conditions in the LCL filter
        init = self.lcl_filter.emt_init
        i_bus_d, i_bus_q = init.i_bus_d, init.i_bus_q
        v_bus_d, v_bus_q = init.v_bus_d, init.v_bus_q
        i_bus_D, i_bus_Q = init.i_bus_D, init.i_bus_Q
        # Current controller initial conditions
        z_cc_d, z_cc_q = self.current_controller.emt_init.z_cc_d, self.current_controller.emt_init.z_cc_q

        # Create each components quadratic bilinear model
        # Phase locked loop
        pll_qbm = self.phase_locked_loop.get_quadratic_bilinear_model(
            v_mag = v_mag, 
            relative_phase_deg = phase_deg
            )
        # Power controller
        apc_qbm = self.active_power_controller.get_quadratic_bilinear_model(
            z_apc = i_bus_d, 
            p_ref = p_bus, 
            p = p_bus
            )
        rpc_qbm = self.reactive_power_controller.get_quadratic_bilinear_model(
            z_rpc = i_bus_q, 
            q_ref = q_bus, 
            q = q_bus
            )
        # Inner current controller
        cc_qbm = self.current_controller.get_small_signal_model(
            z_cc_d = z_cc_d, 
            z_cc_q = z_cc_q, 
            i_d = i_bus_d, 
            i_q = i_bus_q, 
            v_d = v_bus_d, 
            v_q = v_bus_q,
            w = 1
            )
        # Convert to a QBM model and set initial conditions to zero
        cc_qbm = cc_qbm.to_quadratic_bilinear()
        cc_qbm.x.init *= 0
        cc_qbm.y.init *= 0
        cc_qbm.u.init *= 0
        # LCL filter
        br1_qbm = self.lcl_br1.get_quadratic_bilinear_model(
            v_from_d = init.v_vsc_d, 
            v_from_q = init.v_vsc_q, 
            v_to_d = init.v_sh_d, 
            v_to_q = init.v_sh_q,
            i_d = init.i_vsc_d, 
            i_q = init.i_vsc_q,
            name = "vsc"
            )
        br2_qbm = self.lcl_br2.get_quadratic_bilinear_model(
            v_from_d = init.v_sh_D, 
            v_from_q = init.v_sh_Q,
            v_to_d = init.v_bus_D, 
            v_to_q = init.v_bus_Q,
            i_d = init.i_bus_D, 
            i_q = init.i_bus_Q,
            name = "bus"
        )
        sh_qbm = self.lcl_sh.get_quadratic_bilinear_model(
            v_d = init.v_sh_D, 
            v_q = init.v_sh_Q, 
            i_d = (init.i_vsc_D - init.i_bus_D), 
            i_q = (init.i_vsc_Q - init.i_bus_Q) 
        )

        # Inverter level inputs and outputs
        v_bus_D, v_bus_Q = init.v_bus_D, init.v_bus_Q
        i_bus_D, i_bus_Q = init.i_bus_D, init.i_bus_Q
        u = DynamicalVariables(
            name=["p_ref", "q_ref", "w_set", "w_slack", "one", "v_bus_D", "v_bus_Q"],
            type=["device", "device", "device", "device", "device", "grid", "grid"],
            init=[p_bus, q_bus, 1, 1, 1, v_bus_D, v_bus_Q])
        y = DynamicalVariables(
            name=['i_bus_D', 'i_bus_Q'],
            init=[i_bus_D, i_bus_Q])

        # Generate small-signal model
        components = [pll_qbm, apc_qbm, rpc_qbm, cc_qbm, br1_qbm, br2_qbm, sh_qbm]
        
        i_ref_dq = np.array([[init.i_bus_d],[init.i_bus_q]])
        v_bus_dq = np.array([[init.v_bus_d],[init.v_bus_q]])
        i_bus_dq = np.array([[init.i_bus_d],[init.i_bus_q]])
        v_vsc_dq = np.array([[init.v_vsc_d],[init.v_vsc_q]])
        
        connections = self.get_interconnections_qbm(i_ref_dq, i_bus_dq, v_bus_dq, v_vsc_dq)
        self.qbm = QuadraticBilinearModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")
        return self.qbm

    def get_interconnections_qbm(self, i_ref_dq, i_bus_dq, v_bus_dq, v_vsc_dq):
        """
        Linear Interconnections
        -----------------------

        ┌ component ──▶           │ PLL         ┆ APC     ┆ RPC     ┆ ICC      ┆ RL_1      RL_2      RC      │ Grid inputs
        │       ┌ index ──▶       │ 0   1   2   ┆ 3       ┆ 4       ┆ 5,6      ┆ 7,8       9,10      11,12   │ 0      1      2      3        4    5,6
        ▼       ▼                 │ ω   sin cos ┆ i_ref_d ┆ i_ref_q ┆Δv_vsc_dq ┆ i_vsc_dq  i_bus_DQ  v_sh_DQ │ p_ref  q_ref  ω_set  ω_slack  one  v_bus_DQ
        ──────────────────────────┼─────────────┴─────────┴─────────┴──────────┴─────────────────────────────┼────────────────────────────────────────────
        PLL     0        ω_set    │ 0   0   0     0         0         0          0          0        0       │ 0      0      1      0        0     0
                1        ω_slack  │ 0   0   0     0         0         0          0          0        0       │ 0      0      0      1        0     0
                2        one      │ 0   0   0     0         0         0          0          0        0       │ 0      0      0      0        1     0
                3,4      v_bus_DQ │ 0   0   0     0         0         0          0          0        0       │ 0      0      0      0        0     I₂        
        APC     5        p_ref    │ 0   0   0     0         0         0          0          0        0       │ 1      0      0      0        0     0
                6       *p_bus    │ 0   0   0     0         0         0          0          0        0       │ 0      0      0      0        0     0
        RPC     7        q_ref    │ 0   0   0     0         0         0          0          0        0       │ 0      1      0      0        0     0
                8       *q_bus    │ 0   0   0     0         0         0          0          0        0       │ 0      0      0      0        0     0
        ICC     9,10     Δi_ref_dq│ 0   0   0          I₂             0          0          0        0       │ 0      0      0      0   -i_ref_dq  0
                11,12   *Δi_bus_dq│ 0   0   0     0         0         0          0          0        0       │ 0      0      0      0   -i_bus_dq  0
                13,14   *Δv_bus_dq│ 0   0   0     0         0         0          0          0        0       │ 0      0      0      0   -v_bus_dq  0
                15       Δω       │ 1   0   0     0         0         0          0          0        0       │ 0      0      0      0       -1     0
        RL_1    16       ω        │ 1   0   0     0         0         0          0          0        0       │ 0      0      0      0        0     0
                17,18    v_vsc_dq │ 0   0   0     0         0         I₂         0          0        0       │ 0      0      0      0   +v_vsc_dq  0
                19,20   *v_sh_dq  │ 0   0   0     0         0         0          0          0        0       │ 0      0      0      0        0     0
        RL_2    21       ω_slack  │ 0   0   0     0         0         0          0          0        0       │ 0      0      0      1        0     0
                22,23    v_sh_DQ  │ 0   0   0     0         0         0          0          0        I₂      │ 0      0      0      0        0     0
                24,25    v_bus_DQ │ 0   0   0     0         0         0          0          0        0       │ 0      0      0      0        0     I₂
        RC      26       ω_slack  │ 0   0   0     0         0         0          0          0        0       │ 0      0      0      1        0     0
                27,28   *i_sh_DQ  │ 0   0   0     0         0         0          0         -I₂       0       │ 0      0      0      0        0     0
        ──────────────────────────┼──────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────
        Grid    0,1      i_bus_DQ │ 0   0   0     0         0          0         0          I₂         0     │ 0      0      0      0        0     0
        outputs                  

        idx_11 = [([9,10],[3,4],I), ([15],[0],1), ([16],[0],1), ([17,18],[5,6],I), ([22,23],[11,12],I), ([27,28],[9,10],-I)]
        idx_12 = [
            ([0,1,2,3,4],[2,3,4,5,6], np.eye(5)), ([5],[0],1), ([7],[1],1), ([9,10],[4],-i_ref_dq), ([11,12],[4],-i_bus_dq), 
            ([13,14],[4],-v_bus_dq), ([15],[4],-1),([17,18],[4],v_vsc_dq), ([21],[3],1), ([24,25],[5,6], I), ([26],[3],1)
        ]

        
        Nonlinear Interconnections
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

                            2     │ 0,1   2   3   ┆ 4,5,6,7 ┆ 8,9       10,11     12,13
        (x_2 * x)           sin * │ ...   sin cos ┆ ...     ┆ i_vsc_dq  i_bus_DQ  v_sh_DQ
        ──────────────────────────┼───────────────┴─────────┴───────────────────────────────────    
        ICC     11,12   *i_bus_dq │ 0     0   0     0         0         J₂         0
        RL_1    19,20   *v_sh_dq  │ 0     0   0     0         0         0          J₂
        RC      27,28   *i_sh_DQ  │ 0     0   0     0        -J₂        0          0

                            3     │ 0,1   2   3   ┆ 4,5,6,7 ┆ 8,9       10,11     12,13
        (x_3 * x)           cos * │ ...   sin cos ┆ ...     ┆ i_vsc_dq  i_bus_DQ  v_sh_DQ
        ──────────────────────────┼───────────────┴─────────┴───────────────────────────────────    
        ICC     11,12   *i_bus_dq │ 0     0   0     0         0         I₂         0
        RL_1    19,20   *v_sh_dq  │ 0     0   0     0         0         0          I₂
        RC      27,28   *i_sh_DQ  │ 0     0   0     0         I₂        0          0
                            
                        3         │ 0,1   2   3   ┆ 4,5  ┆ 6,7  ┆ 8,9      10      11      12,13
        (u_5 * x)       v_bus_D * │ z_ab  sin cos ┆ z_dq ┆ z_cc ┆ i_vsc_dq i_bus_D i_bus_Q  v_sh_DQ
        ──────────────────────────┼───────────────┴──────┴──────┴───────────────────────────────────    
        APC     6       *p_bus    │ 0     0   0     0      0      0        1       0        0       
        RPC     8       *q_bus    │ 0     0   0     0      0      0        0      -1        0
        ICC     13      *v_bus_d  │ 0     0   1     0      0      0        0       0        0
                14      *v_bus_q  │ 0    -1   0     0      0      0        0       0        0
        
                        4         │ 0,1   2   3   ┆ 4,5  ┆ 6,7  ┆ 8,9      10      11      12,13
        (u_6 * x)       v_bus_Q * │ z_ab  sin cos ┆ z_dq ┆ z_cc ┆ i_vsc_dq i_bus_D i_bus_Q  v_sh_DQ
        ──────────────────────────┼───────────────┴──────┴──────┴───────────────────────────────────    
        APC     6       *p_bus    │ 0     0   0     0      0      0        0       1        0       
        RPC     8       *q_bus    │ 0     0   0     0      0      0        1       0        0
        ICC     13      *v_bus_d  │ 0     1   0     0      0      0        0       0        0
                14      *v_bus_q  │ 0     0   1     0      0      0        0       0        0

        idx_x2 = [([11,12],[10,11],J), ([19,20],[12,13],J), ([27,28],[8,9],J.T)]
        idx_x3 = [([11,12],[10,11],I), ([19,20],[12,13],I), ([27,28],[8,9],I)]
        
        idx_u5 = [([6], [10], 1), ([8], [11],-1), ([13,14], [2,3], J)]
        idx_u6 = [([6], [11], 1), ([8], [10], 1), ([13,14], [2,3], I)]
        """
        # Matrix values
        I = np.eye(2)
        J = np.array([[0, 1], [-1,0]])

        # Number of stacked/grid side inputs and outputs
        u_stack = 29
        y_stack = 13
        x_stack = 14
        u_grid = 7
        y_grid = 2

        # Matrix data in (row, column, value) format
        idx_11 = [([9,10],[3,4],I), ([15],[0],1), ([16],[0],1), ([17,18],[5,6],I), ([22,23],[11,12],I), ([27,28],[9,10],-I)]
        idx_12 = [
            ([0,1,2,3,4],[2,3,4,5,6], np.eye(5)), ([5],[0],1), ([7],[1],1), ([9,10],[4],-i_ref_dq), ([11,12],[4],-i_bus_dq), 
            ([13,14],[4],-v_bus_dq), ([15],[4],-1),([17,18],[4],v_vsc_dq), ([21],[3],1), ([24,25],[5,6], I), ([26],[3],1)
        ]

        idx_x2 = [([11,12],[10,11],J), ([19,20],[12,13],J), ([27,28],[8,9],J.T)]
        idx_x3 = [([11,12],[10,11],I), ([19,20],[12,13],I), ([27,28],[8,9],I)]
        
        idx_u5 = [([6],[10],1), ([8],[11],-1), ([13,14],[2,3],J)]
        idx_u6 = [([6],[11],1), ([8],[10],1), ([13,14],[2,3],I)]


        # Linear interconnection matrices
        L11 = coordinates_to_matrix(shape=(u_stack, y_stack), data=idx_11)
        L12 = coordinates_to_matrix(shape=(u_stack, u_grid), data=idx_12)
        L21 = coordinates_to_matrix(shape=(y_grid, y_stack), data=[([0,1],[9,10],I)])
        L22 = np.zeros((y_grid, u_grid))

        # Nonlinear interconnection matrices
        M1_x2 = coordinates_to_matrix(shape=(u_stack, x_stack), data=idx_x2)
        M1_x3 = coordinates_to_matrix(shape=(u_stack, x_stack), data=idx_x3)
        M2_u5 = coordinates_to_matrix(shape=(u_stack, x_stack), data=idx_u5)
        M2_u6 = coordinates_to_matrix(shape=(u_stack, x_stack), data=idx_u6)

        Z = np.zeros((u_stack, x_stack))
        M1 = np.hstack([Z, Z, M1_x2, M1_x3] + 10*[Z])
        M2 = np.hstack(5*[Z] + [M2_u5, M2_u6])
        
        return (L11, L12, L21, L22, M1, M2)


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

    def get_derivative_state_emt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        It returns a vector with the differential equations that describe the dynamics of the GFLI.
        This model includes: pi controller, pll, and LCL filter.
        """     
        # Unpack states
        (v_pll_q, z_pll, theta_pll, z_apc, z_rpc, z_cc_d, z_cc_q,
        i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c) = x
        # Unpack *external* inputs
        p_ref, q_ref, v_bus_a, v_bus_b, v_bus_c = u

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


    def get_output_emt(self, x: np.ndarray) -> np.ndarray:
            (v_pll_q, z_pll, theta_pll, z_apc, z_rpc, z_cc_d, z_cc_q,
                    i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c) = x
                
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

        # Compute power
        p_sh = [v_d * i_d + v_q * i_q for v_d, v_q, i_d, i_q in zip(v_sh_d, v_sh_q, i_bus_d, i_bus_q)]
        q_sh = [v_q * i_d - v_d * i_q for v_d, v_q, i_d, i_q in zip(v_sh_d, v_sh_q, i_bus_d, i_bus_q)]

        results = DynamicalVariables(
            name = ['v_pll_q', 'z_pll', 'theta_pll', 'z_apc', 'z_rpc',  'z_cc_d', 'z_cc_q', "i_vsc_d", "i_vsc_q", "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q", "p_sh", "q_sh"],
            component = f"{self.type_}_{self.id}",
            value=[v_pll_q, z_pll, theta_pll, z_apc, z_rpc, z_cc_d, z_cc_q, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q, p_sh, q_sh],
            time=self.variables_emt.x.time
        )

        return results