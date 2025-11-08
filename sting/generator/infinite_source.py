import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from collections import namedtuple

from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from sting.utils.linear_systems_tools import State_space_model

class Power_flow_variables(NamedTuple):
    p_bus: float = 0
    q_bus: float = 0
    vmag_bus: float = 0
    vphase_bus: float = 0

class EMT_initial_conditions(NamedTuple):
    vmag_bus: float = 0
    vphase_bus: float = 0
    p_bus: float = 0
    q_bus: float = 0
    v_bus_DQ: float = 0
    i_bus_DQ: complex = 0
    p_int: float = 0
    q_int: float = 0
    ref_angle: float = 0
    x: np.ndarray = 0
    y: np.ndarray = 0

@dataclass(slots=True)
class Infinite_source:
    idx: str
    bus_idx: str 
    p_min: float
    p_max: float
    q_min: float
    q_max: float
    sbase: float
    vbase: float
    fbase: float
    r: float
    l: float
    pf: Power_flow_variables = field(default_factory=Power_flow_variables)
    emt_init_cond: EMT_initial_conditions = field(default_factory=EMT_initial_conditions)
    ssm: Optional[State_space_model] = None
    name: str = field(default_factory=str)
    type: str = 'generator'

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.generators.loc[self.idx]
        self.pf  = Power_flow_variables(p_bus = sol.p.item(), 
                                        q_bus = sol.q.item(), 
                                        vmag_bus = sol.bus_vmag.item(),
                                        vphase_bus = sol.bus_vphase.item())

    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.pf.vmag_bus
        vphase_bus = self.pf.vphase_bus
        p_bus = self.pf.p_bus
        q_bus = self.pf.q_bus
        
        v_bus_DQ = vmag_bus*np.exp(vphase_bus*1j*np.pi/180)
        i_bus_DQ = ((p_bus + 1j*q_bus)/v_bus_DQ).conjugate()
        i_bus_D, i_bus_Q = np.real(i_bus_DQ), np.imag(i_bus_DQ)

        v_int_DQ = v_bus_DQ + i_bus_DQ*(self.r + 1j*self.l)
        v_int_DQ_angle = np.angle(v_int_DQ, deg=True)
        ref_angle = v_int_DQ_angle

        p_int = np.real(v_int_DQ*np.conjugate(v_int_DQ_angle))
        q_int = np.imag(v_int_DQ*np.conjugate(v_int_DQ_angle))

        v_bus_dq = v_bus_DQ*np.exp(-ref_angle*np.pi/180*1j)
        v_bus_d, v_bus_q = np.real(v_bus_dq), np.imag(v_bus_dq)
        v_int_dq =  v_int_DQ*np.exp(-ref_angle*np.pi/180*1j)
        v_int_d, v_int_q  = np.real(v_int_dq), np.imag(v_int_dq)

        i_bus_dq =  i_bus_DQ*np.exp(-ref_angle*np.pi/180*1j)
        i_bus_d, i_bus_q  = np.real(i_bus_dq), np.imag(i_bus_dq)

        x = np.array([ i_bus_d, i_bus_q])
        y = np.array([ i_bus_D, i_bus_Q])

        local_vars = locals()
        data = {field: local_vars[field] for field in EMT_initial_conditions._fields}

        self.emt_init_cond = EMT_initial_conditions(**data)

    def _build_small_signal_model(self):

        r = self.r
        l = self.l
        wb = 2*np.pi*self.fbase
        cosphi = np.cos(self.emt_init_cond.ref_angle*np.pi/180)
        sinphi = np.sin(self.emt_init_cond.ref_angle*np.pi/180)
        
        Rotmat = np.array([[cosphi, -sinphi], 
                           [sinphi,  cosphi]])
    
        A = wb*np.array([   [-r/l, 1],
                            [-1, -r/l]  ])
    
        B = (wb*np.array([[1/l, 0,-1/l, 0],
                          [0, 1/l, 0, -1/l]])) @ block_diag(np.eye(2), np.transpose(Rotmat))

        C = Rotmat

        D = np.zeros((2,4))

        self.ssm = State_space_model(A = A,
                                     B = B,
                                     C = C,
                                     D = D,
                                     states= ["i2c_d", "i2c_q"],
                                     grid_side_inputs= ["v2_d", "v2_q"],
                                     device_side_inputs=["v1c_d", "v1c_q"],
                                     outputs=["i2_d", "i2_q"])

