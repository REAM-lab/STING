# Import standard python packages and third-party packages
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import copy

from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

class Power_flow_variables(NamedTuple):
    vmag_bus: float 
    vphase_bus: float 

class EMT_initial_conditions(NamedTuple):
    vmag_bus: float 
    vphase_bus: float 
    v_bus_D: float
    v_bus_Q: float
    i_bus_D: float
    i_bus_Q: float

@dataclass
class Parallel_rc_shunt:
    idx: int
    bus_idx: int
    sbase: float	
    vbase: float
    fbase: float
    r: float
    c: float
    pf: Optional[Power_flow_variables] = None
    emt_init_cond: Optional[EMT_initial_conditions] = None
    ssm: Optional[StateSpaceModel] = None
    name: str = field(default_factory=str)
    type: str = 'pa_rc'

    @property
    def g(self):
        return 1/self.r
    
    @property
    def b(self):
        return 1/self.c

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.shunts.loc[f"{self.type}_{self.idx}"]
        self.pf  = Power_flow_variables(vmag_bus = sol.bus_vmag.item(),
                                        vphase_bus = sol.bus_vphase.item())

    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.pf.vmag_bus
        vphase_bus = self.pf.vphase_bus

        v_bus_DQ = vmag_bus*np.exp(vphase_bus*1j*np.pi/180);      
        i_bus_DQ =  v_bus_DQ*self.g +  v_bus_DQ*(1j*self.b)

        self.emt_init_cond = EMT_initial_conditions(vmag_bus = vmag_bus,
                                                    vphase_bus = vphase_bus,
                                                    v_bus_D = v_bus_DQ.real,
                                                    v_bus_Q = v_bus_DQ.imag,
                                                    i_bus_D = i_bus_DQ.real,
                                                    i_bus_Q = i_bus_DQ.imag)

    def _build_small_signal_model(self):
        g = self.g
        b = self.b
        wb = 2*np.pi*self.fbase
    
        # Define state-space matrices
        A = wb*np.array([[-g/b, 1],
                         [-1, -g/b]])

        B = wb*np.array([[1/b, 0],
                         [0,    1/b]])

        C = np.eye(2)

        D = np.zeros((2,2))      

        u = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            component=[f"{self.type}_{self.idx}"]*2,
            type=["grid"]*2,
            init=[self.emt_init_cond.i_bus_D, self.emt_init_cond.i_bus_Q]
        )
        
        x = DynamicalVariables(
            name=["v_bus_D", "v_bus_Q"],
            component=[f"{self.type}_{self.idx}"]*2,
            type=["grid"]*2,
            init=[self.emt_init_cond.v_bus_D, self.emt_init_cond.v_bus_Q]
        )
        y = copy.deepcopy(x)

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)
        