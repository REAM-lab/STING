"""
This module contains the GFMI generator that includes:
- Virtual inertia control
- Droop control for reactive power
- LCL filter
- Voltage magnitude controller

There is no DC-side dynamics modeled for the GFMI.
"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from typing import NamedTuple, Optional, ClassVar
from dataclasses import dataclass, field

# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

# -----------
# Sub-classes
# -----------
class PowerFlowVariables(NamedTuple):
    p_bus: float
    q_bus: float
    vmag_bus: float
    vphase_bus: float


class InitialConditionsEMT(NamedTuple):
    vmag_bus: float
    vphase_bus: float
    p_bus: float
    q_bus: float
    p_ref: float
    q_ref: float
    v_ref: float
    angle_ref: float
    v_vsc_d: float
    i_vsc_d: float
    i_vsc_q: float
    i_bus_d: float
    i_bus_q: float
    v_lcl_sh_d: float
    v_lcl_sh_q: float
    i_bus_D: float
    i_bus_Q: float
    v_bus_D: float
    v_bus_Q: float
    v_vsc_mag: float
    v_vsc_DQ_phase: float

# -----------
# Main class
# -----------
@dataclass(slots=True)
class GFMIc:
    id: int = field(default=-1, init=False)
    bus: str
    p_min: float
    p_max: float
    q_min: float
    q_max: float
    base_power_MVA: float
    base_voltage_kV: float
    base_frequency_Hz: float
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
    h_s: float
    kd_pu: float
    droop_q_pu: float
    tau_pc_s: float
    kp_vc_pu: float
    ki_vc_puHz: float
    v_dc_pu: float
    bus_id: int = None
    name: str = field(default_factory=str)
    type: str = "gfmi_c"
    pf: Optional[PowerFlowVariables] = None
    emt_init: Optional[InitialConditionsEMT] = None
    ssm: Optional[StateSpaceModel] = None
    tags: ClassVar[list[str]] = ["generator"]

    @property
    def rf2_pu(self):
        return (self.txr_r1_pu + self.txr_r2_pu) * self.base_power_MVA / self.txr_power_MVA

    @property
    def xf2_pu(self):
        return (self.txr_x1_pu + self.txr_x2_pu) * self.base_power_MVA / self.txr_power_MVA
    
    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz
    
    def post_system_init(self, system):
        self.bus_id = next((n for n in system.bus if n.name == self.bus)).id

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.generators.loc[f"{self.type}_{self.id}"]
        self.pf = PowerFlowVariables(
            p_bus=sol.p.item(),
            q_bus=sol.q.item(),
            vmag_bus=sol.bus_vmag.item(),
            vphase_bus=sol.bus_vphase.item(),
        )

    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.pf.vmag_bus
        vphase_bus = self.pf.vphase_bus
        p_bus = self.pf.p_bus
        q_bus = self.pf.q_bus

        # Voltage in the end of the LCL filter
        v_bus_DQ = vmag_bus * np.exp(vphase_bus * np.pi / 180 * 1j)

        # Current sent from the end of the LCL filter
        i_bus_DQ = (p_bus - q_bus * 1j) / np.conjugate(v_bus_DQ)

        # Voltage across the shunt element in the LCL filter
        v_lcl_sh_DQ = v_bus_DQ + (self.rf2_pu + self.xf2_pu * 1j) * i_bus_DQ

        # Voltage and power references
        v_ref = abs(v_lcl_sh_DQ)
        s_ref = v_lcl_sh_DQ * np.conjugate(i_bus_DQ)
        p_ref = s_ref.real
        q_ref = s_ref.imag

        # Current flowing through shunt element of LCL filter
        i_lcl_sh_DQ = v_lcl_sh_DQ * (self.csh_pu * 1j) + v_lcl_sh_DQ / self.rsh_pu

        # Current sent from the beginning of the LCL filter
        i_vsc_DQ = i_bus_DQ + i_lcl_sh_DQ
        v_vsc_DQ = v_lcl_sh_DQ + (self.rf1_pu + self.xf1_pu * 1j) * i_vsc_DQ
        
        # Angle reference
        angle_ref = np.angle(v_vsc_DQ, deg=True)

        # We refer the voltage and currents to the synchronous frames of the
        # inverter
        v_vsc_dq = v_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_vsc_dq = i_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        v_bus_dq = v_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_bus_dq = i_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        v_lcl_sh_dq = v_lcl_sh_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        self.emt_init = InitialConditionsEMT(
            vmag_bus=vmag_bus,
            vphase_bus=vphase_bus,
            p_bus=p_bus,
            q_bus=q_bus,
            p_ref=p_ref,
            q_ref=q_ref,
            v_ref=v_ref,
            angle_ref=angle_ref,
            v_vsc_d=v_vsc_dq.real,
            i_vsc_d=i_vsc_dq.real,
            i_vsc_q=i_vsc_dq.imag,
            i_bus_d=i_bus_dq.real,
            i_bus_q=i_bus_dq.imag,
            v_lcl_sh_d=v_lcl_sh_dq.real,
            v_lcl_sh_q=v_lcl_sh_dq.imag,
            i_bus_D=i_bus_DQ.real,
            i_bus_Q=i_bus_DQ.imag,
            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag,
            v_vsc_mag = abs(v_vsc_DQ),
            v_vsc_DQ_phase = np.angle(v_vsc_DQ, deg=True)
        )

    def _build_small_signal_model(self):
        
        # Power controller (Virtual inertia and droop control for reactive power)
        tau_pc = self.tau_pc_s
        wb = self.wbase
        h = self.h_s
        kd = self.kd_pu
        droop_q = self.droop_q_pu
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q
        v_lcl_sh_d, v_lcl_sh_q = self.emt_init.v_lcl_sh_d, self.emt_init.v_lcl_sh_q
        p_ref, q_ref = self.emt_init.p_ref, self.emt_init.q_ref
        
        pc_controller = StateSpaceModel( 
                                        A = np.array([  [0, wb,           0,          0],
                                                        [0, -kd/(2*h),    -1/(2*h),   0],
                                                        [0, 0,            -1/tau_pc,  0],
                                                        [0, 0,            0,          -1/tau_pc]]),
                                        B = np.vstack(( [0, 0, 0, 0, 0,       0, 0],
                                                        [0, 0, 0, 0, 1/(2*h), 0, 0],
                                                        1/tau_pc*np.array([i_bus_d, i_bus_q, v_lcl_sh_d, v_lcl_sh_q,   0, 0, 0]),
                                                        1/tau_pc*np.array([-i_bus_q, i_bus_d, v_lcl_sh_q, -v_lcl_sh_d, 0, 0, 0]))),
                                        C = np.array([  [1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 0, -droop_q]]),
                                        D = np.vstack(( np.zeros((2,7)),
                                                        np.hstack((np.zeros((5, )), [droop_q, 1])))),
                                        x = DynamicalVariables(name=['angle_pc', 'w_pc', 'p_pc', 'q_pc'],
                                                               init = [0, 0, p_ref, q_ref]),
                                        u = DynamicalVariables(name=['v_lcl_sh_d', 'v_lcl_sh_q', 'i_bus_d', 'i_bus_q', 'p_ref', 'q_ref', 'v_ref']),
                                        y = DynamicalVariables(name=['phi_pc', 'w_pc', 'v_lcl_sh_ref'])
                                        )


        # Voltage magnitude controller
        kp_vc, ki_vc = self.kp_vc_pu, self.ki_vc_puHz
        v_vsc_d = self.emt_init.v_vsc_d
        
        voltage_mag_controller = StateSpaceModel(  
                                                A = np.array([ [0] ]),
                                                B = ki_vc*np.array([[1, -1]]),
                                                C = np.array([[1], [0]]),
                                                D = kp_vc*np.array([[1, -1],
                                                                    [0, 0 ]]),
                                                x = DynamicalVariables(name = ['pi_vc'],
                                                                       init = [v_vsc_d]),
                                                u = DynamicalVariables(name=['v_sh_mag_ref', 'v_sh_mag']),
                                                y =  DynamicalVariables(name=['v_vsc_d', 'v_vsc_q'])
                                                )


        # LCL filter
        rf1, xf1, rf2, xf2, rsh, csh = self.rf1_pu, self.xf1_pu, self.rf2_pu, self.xf2_pu, self.rsh_pu, self.csh_pu
        wb = self.wbase
        i_vsc_d, i_vsc_q = self.emt_init.i_vsc_d, self.emt_init.i_vsc_q
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q
        v_lcl_sh_d, v_lcl_sh_q = self.emt_init.v_lcl_sh_d, self.emt_init.v_lcl_sh_q

        lcl_filter = StateSpaceModel(
                        A = wb*np.array([[-rf1/xf1  ,   1       ,  0        ,   0       ,       -1/xf1      ,  0],
                                         [-1        ,   -rf1/xf1,  0        ,   0       ,       0           ,  -1/xf1],
                                         [0         ,   0       ,  -rf2/xf2 ,   1       ,       1/xf2       ,  0],
                                         [0         ,   0       ,  -1       ,   -rf2/xf2,       0           ,  1/xf2],
                                         [1/csh     ,   0       ,  -1/csh   ,   0       ,       -1/(rsh*csh),  1],
                                         [0         ,   1/csh   ,  0        ,   -1/csh  ,       -1          ,  -1/(rsh*csh)]]),
                        B = wb*np.array([[1/xf1 ,    0      ,   0       ,   0      ,      i_vsc_q],
                                         [0     ,    1/xf1  ,   0       ,   0      ,      -i_vsc_d],
                                         [0     ,    0      ,   -1/xf2  ,   0      ,      i_bus_q],
                                         [0     ,    0      ,   0       ,   -1/xf2 ,      -i_bus_d],
                                         [0     ,    0      ,   0       ,   0      ,      v_lcl_sh_q],
                                         [0     ,    0      ,   0       ,   0      ,      -v_lcl_sh_d]]),
                        C = np.eye(6),
                        D = np.zeros((6,5)),
                        x = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"],
                                               init=[i_vsc_d, i_vsc_q, i_bus_d, i_bus_q, v_lcl_sh_d, v_lcl_sh_q]),
                        u = DynamicalVariables(name=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q', 'w']),
                        y = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"]))
        
        # Construccion of CCM matrices
        v_ref = self.emt_init.v_ref
        angle_ref = self.emt_init.angle_ref
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        sinphi = np.sin(angle_ref*np.pi/180)
        cosphi = np.cos(angle_ref*np.pi/180)
        a = v_lcl_sh_d/v_ref
        b = v_lcl_sh_q/v_ref
        c = -sinphi*v_bus_D+cosphi*v_bus_Q
        d = -cosphi*v_bus_D-sinphi*v_bus_Q
        e = -sinphi*i_bus_d- cosphi*i_bus_q
        f = cosphi*i_bus_d - sinphi*i_bus_q
        
        Fccm = np.vstack((  np.hstack((np.zeros((2,9)), np.eye(2))), # v_lcl_sh_dq
                            np.hstack((np.zeros((2,7)), np.eye(2), np.zeros((2,2)))), # i_bus_dq
                            np.zeros((3,11)), # p_ref, q_ref, v_ref
                            np.hstack( ( [0, 0, 1], np.zeros((8, )) )), # v_lcl_sh_ref
                            np.hstack( ( np.zeros((9, )),  [a, b])), # v_lcl_sh_mag
                            np.hstack( ( np.zeros((2,3)), np.eye(2), np.zeros((2,6))) ), # v_vsc_dq
                            np.hstack( ( [c], np.zeros((10, )) )), # v_bus_d
                            np.hstack( ( [d], np.zeros((10, )) )), # v_bus_q
                            np.hstack( ( [0, 1], np.zeros((9, )))) # w
                            ))
        Gccm = np.vstack(( np.zeros((4,5)) ,
                           np.hstack( (np.eye(3), np.zeros((3,2)))),
                           np.zeros((4,5)),
                           np.hstack( (np.zeros((3,)), [cosphi, sinphi])),
                           np.hstack( (np.zeros((3,)), [-sinphi, cosphi])),
                           np.zeros((5, ))))
        Hccm = np.vstack(( np.hstack(( [e], np.zeros((6,)), [cosphi, -sinphi], [0, 0] )), 
                           np.hstack(( [f], np.zeros((6,)), [sinphi, cosphi], [0, 0] )) ))
        Lccm = np.zeros((2,5))
    
        components = [pc_controller, voltage_mag_controller, lcl_filter]
        connections = [Fccm, Gccm, Hccm, Lccm]

        # Inputs and outputs
        u = DynamicalVariables(
                                    name=["p_ref", "q_ref", "v_ref", "v_bus_D", "v_bus_Q"],
                                    type=["device", "device", "device", "grid", "grid"],
                                    init=[p_ref, q_ref, v_ref, v_bus_D, v_bus_Q]
                                    )

        i_bus_D, i_bus_Q = self.emt_init.i_bus_D, self.emt_init.i_bus_Q
        y = DynamicalVariables(
                                    name=["i_bus_D", "i_bus_Q"],
                                    init=[i_bus_D, i_bus_Q]
                                    )

        # Generate small-signal model
        ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type}_{self.id}")

        self.ssm = ssm
