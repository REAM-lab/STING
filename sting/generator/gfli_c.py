"""
Untested, still in development.
"""

# Import standard python packages
import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

# Import src packages
from sting.utils import linear_systems_tools
from sting.utils.linear_systems_tools import State_space_model


class Power_flow_variables(NamedTuple):
    p_bus: float
    q_bus: float
    vmag_bus: float
    vphase_bus: float


class EMT_initial_conditions(NamedTuple):
    vmag_bus: float
    vphase_bus: float
    p_bus: float
    q_bus: float
    angle_ref: float
    pi_cc_d: float
    pi_cc_q: float
    i_bus_d: float
    i_bus_q: float
    i_bus_D: float
    i_bus_Q: float
    v_bus_D: float
    v_bus_Q: float
    v_vsc_mag: float
    v_vsc_DQ_phase: float


@dataclass(slots=True)
class GFLI_c:
    """Grid-following inverter that has L filter and PI current controller. No PLL and No DC-side
    dynamics are included."""

    idx: str
    bus_idx: str
    p_min: float
    p_max: float
    q_min: float
    q_max: float
    sbase: float
    vbase: float
    vbase: float
    fbase: float
    v_dc: float
    rf: float
    lf: float
    txr_sbase: float
    txr_r1: float
    txr_l1: float
    txr_r2: float
    txr_l2: float
    beta: float
    kp_cc: float
    ki_cc: float
    name: str = field(default_factory=str)
    type: str = "generator"
    pf: Optional[Power_flow_variables] = None
    emt_init_cond: Optional[EMT_initial_conditions] = None
    ssm: Optional[State_space_model] = None

    @property
    def txr_r(self):
        return (self.txr_r1 + self.txr_r2) * self.sbase / self.txr_sbase

    @property
    def txr_l(self):
        return (self.txr_l1 + self.txr_l2) * self.sbase / self.txr_sbase

    @property
    def wbase(self):
        return 2 * np.pi * self.fbase

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.generators.loc[self.idx]
        self.pf = Power_flow_variables(
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

        # Voltage in the end of the L filter
        v_bus_DQ = vmag_bus * np.exp(vphase_bus * np.pi / 180 * 1j)
        ref_angle = np.angle(v_bus_DQ, deg=True)

        # Current sent from the end of the L filter
        i_bus_DQ = (p_bus - q_bus * 1j) / np.conjugate(v_bus_DQ)

        # Voltage at vsc
        v_vsc_DQ = (
            v_bus_DQ + (self.rf + self.txr_r + (self.lf + self.txr_l) * 1j) * i_bus_DQ
        )

        # We refer the voltage and currents to the synchronous frames of the
        # inverter
        v_vsc_dq = v_vsc_DQ * np.exp(-ref_angle * np.pi / 180 * 1j)
        v_bus_dq = v_bus_DQ * np.exp(-ref_angle * np.pi / 180 * 1j)
        i_bus_dq = i_bus_DQ * np.exp(-ref_angle * np.pi / 180 * 1j)

        # Initial conditions for the integral controller
        pi_cc_dq = (
            v_vsc_dq - 1j * (self.lf + self.txr_l) * i_bus_dq - self.beta * v_bus_dq
        )

        self.emt_init_cond = EMT_initial_conditions(
            vmag_bus=vmag_bus,
            vphase_bus=vphase_bus,
            p_bus=p_bus,
            q_bus=q_bus,
            angle_ref=ref_angle,
            pi_cc_d=pi_cc_dq.real,
            pi_cc_q=pi_cc_dq.imag,
            i_bus_d=i_bus_dq.real,
            i_bus_q=i_bus_dq.imag,
            i_bus_D=i_bus_DQ.real,
            i_bus_Q=i_bus_DQ.imag,
            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag,
            v_vsc_mag=abs(v_vsc_DQ),
            v_vsc_DQ_phase=np.angle(v_vsc_DQ, deg=True),
        )

    def _build_small_signal_model(self):

        # Current PI controller
        kp_cc, ki_cc = self.kp_cc, self.ki_cc
        pi_cc_d, pi_cc_q = self.emt_init_cond.pi_cc_d, self.emt_init_cond.pi_cc_q

        pi_controller = State_space_model(
            A=np.zeros((2, 2)),
            B=ki_cc * np.hstack((np.eye(2), -np.eye(2))),
            C=np.eye(2),
            D=kp_cc * np.hstack((np.eye(2), -np.eye(2))),
            inputs=["i_bus_d_ref", "i_bus_q_ref"],
            states=["pi_cc_d", "pi_cc_q"],
            outputs=["e_d", "e_q"],
            initial_states=np.array([[pi_cc_d], [pi_cc_q]]),
        )

        # L filter
        rf = self.rf + self.txr_r
        lf = self.lf + self.txr_l
        wb = self.wbase
        beta = self.beta
        i_bus_d, i_bus_q = self.emt_init_cond.i_bus_d, self.emt_init_cond.i_bus_q
        sinphi = np.sin(self.emt_init_cond.angle_ref * np.pi / 180)
        cosphi = np.cos(self.emt_init_cond.angle_ref * np.pi / 180)
        # fmt: off
        l_filter = State_space_model( A = wb*np.array([[-rf/lf,  1], 
                                                       [-1    ,  -rf/lf]]),
                                      B = wb*np.array([[ 1/lf ,  0   ,-1/lf  ,0] ,
                                                       [0,      1/lf, 0,    -1/lf]]),
                                      C = np.eye(2),
                                      D = np.zeros((2,4)),
                                      states= ['i_bus_d', 'i_bus_q'],
                                      inputs=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q'],
                                      outputs=['i_bus_d', 'i_bus_q'],
                                      initial_states = np.array([[i_bus_d], [i_bus_q]]))
        
        
        # Interconnection matrices
        Fccm = np.vstack( (     np.zeros((1,4)) ,
                                np.zeros((1,4)) , 
                                np.hstack((np.zeros((2,2)), np.eye(2))), 
                                [1,  0,  0, -lf  ], 
                                [0,  1, lf,   0  ], 
                                np.zeros((2,4))  )) 

        Gccm = np.vstack((      [ 1, 0,  0, 0], 
                                [0,  1, 0, 0], 
                                np.zeros((2,4)),
                                [0, 0, beta*cosphi ,   beta*sinphi],  
                                [0, 0, -beta*sinphi,    beta*cosphi], 
                                [0, 0, cosphi ,sinphi], 
                                [0, 0,  -sinphi ,cosphi], 
                                ) )  
  
        Hccm = np.vstack(( [ 0, 0 ,cosphi , -sinphi],
                            [0, 0, sinphi , cosphi  ] ))
        # fmt: on
        Lccm = np.zeros((2, 4))

        # Generate small-signal model
        ssm = linear_systems_tools.connect_models_via_CCM(
            Fccm, Gccm, Hccm, Lccm, [pi_controller, l_filter]
        )

        # Note that states and initial states do not need to be defined as they result from stacking states in ccm tool.

        # Inputs and outputs
        device_side_inputs = ["i_bus_d_ref", "i_bus_q_ref"]
        initial_device_side_inputs = np.array([[i_bus_d], [i_bus_q]])

        grid_side_inputs = ["v_bus_D", "v_bus_Q"]
        v_bus_D, v_bus_Q = self.emt_init_cond.v_bus_D, self.emt_init_cond.v_bus_Q
        initial_grid_side_inputs = np.array([[v_bus_D], [v_bus_Q]])

        outputs = ["i_bus_D", "i_bus_Q"]
        i_bus_D, i_bus_Q = self.emt_init_cond.i_bus_D, self.emt_init_cond.i_bus_Q
        initial_outputs = np.array([[i_bus_D], [i_bus_Q]])

        self.ssm = State_space_model(
            A=ssm.A,
            B=ssm.B,
            C=ssm.C,
            D=ssm.D,
            states=ssm.states,
            initial_states=ssm.initial_states,
            outputs=outputs,
            initial_outputs=initial_outputs,
            device_side_inputs=device_side_inputs,
            initial_device_side_inputs=initial_device_side_inputs,
            grid_side_inputs=grid_side_inputs,
            initial_grid_side_inputs=initial_grid_side_inputs,
        )
