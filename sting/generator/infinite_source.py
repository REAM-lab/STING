# Import standard python packages and third-party packages
import numpy as np
from scipy.linalg import block_diag
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

# Import sting packages
from sting.models.StateSpaceModel import StateSpaceModel
from sting.models.Variables import Variables

class Power_flow_variables(NamedTuple):
    p_bus: float
    q_bus: float 
    vmag_bus: float 
    vphase_bus: float 

class EMT_initial_conditions(NamedTuple):
    v_bus_D: float
    v_bus_Q: float
    v_int_d: float
    v_int_q: float
    i_bus_d: float
    i_bus_q: float
    i_bus_D: float
    i_bus_Q: float
    ref_angle: float

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
    pf: Optional[Power_flow_variables] = None
    emt_init_cond: Optional[EMT_initial_conditions] = None
    ssm: Optional[StateSpaceModel] = None
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

        v_int_DQ = v_bus_DQ + i_bus_DQ*(self.r + 1j*self.l)
        ref_angle = np.angle(v_int_DQ, deg=True)

        v_int_dq =  v_int_DQ*np.exp(-ref_angle*np.pi/180*1j)
        i_bus_dq =  i_bus_DQ*np.exp(-ref_angle*np.pi/180*1j)

        self.emt_init_cond = EMT_initial_conditions(v_bus_D = v_bus_DQ.real,
                                                    v_bus_Q = v_bus_DQ.imag,
                                                    v_int_d = v_int_dq.real,
                                                    v_int_q = v_int_dq.imag,
                                                    i_bus_d = i_bus_dq.real,
                                                    i_bus_q = i_bus_dq.imag,
                                                    i_bus_D = i_bus_DQ.real,
                                                    i_bus_Q = i_bus_DQ.imag,
                                                    ref_angle=ref_angle
                                                    )

    def _build_small_signal_model(self):

        r = self.r
        l = self.l
        wb = 2*np.pi*self.fbase
        cosphi = np.cos(self.emt_init_cond.ref_angle*np.pi/180)
        sinphi = np.sin(self.emt_init_cond.ref_angle*np.pi/180)
        
        Rotmat = np.array([[cosphi, -sinphi], 
                           [sinphi,  cosphi]])
    
        # State-space matrices
        A = wb*np.array([   [-r/l, 1],
                            [-1, -r/l]  ])
    
        B = (wb*np.array([[1/l, 0,-1/l, 0],
                          [0, 1/l, 0, -1/l]])) @ block_diag(np.eye(2), np.transpose(Rotmat))

        C = Rotmat

        D = np.zeros((2,4))


        # Inputs
        v_bus_D, v_bus_Q = self.emt_init_cond.v_bus_D, self.emt_init_cond.v_bus_Q
        v_int_d, v_int_q = self.emt_init_cond.v_int_d, self.emt_init_cond.v_int_q
        
        u = Variables(
            name=["v_bus_D", "v_bus_Q", "v_ref_d", "v_ref_q"],
            component=[self.idx]*4,
            v_type=["grid", "grid", "device", "device"],
            init=[v_bus_D, v_bus_Q, v_int_d, v_int_q]
        )
        
        # Outputs
        i_bus_D, i_bus_Q = self.emt_init_cond.i_bus_D, self.emt_init_cond.i_bus_Q
        
        y = Variables(
            name=["i_bus_D", "i_bus_Q"],
            component=[self.idx]*2,
            v_type=["grid", "grid"],
            init=[i_bus_D, i_bus_Q]
        )
        
        # States
        i_bus_d, i_bus_q = self.emt_init_cond.i_bus_d, self.emt_init_cond.i_bus_q
        
        x = Variables(
            name=["i_bus_d", "i_bus_q"],
            component=[self.idx]*2,
            v_type=["device", "device"],
            init=[i_bus_d, i_bus_q]
        )
        
        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)

