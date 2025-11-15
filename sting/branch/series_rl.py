# Import standard python packages and third-party packages
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import numpy as np

# Import sting packages
from sting.models.StateSpaceModel import StateSpaceModel
from sting.models.Variables import Variables

class Power_flow_variables(NamedTuple):
    vmag_from_bus: float 
    vphase_from_bus: float 
    vmag_to_bus: float 
    vphase_to_bus: float 

class EMT_initial_conditions(NamedTuple):
    vmag_from_bus: float
    vphase_from_bus: float
    vmag_to_bus: float
    vphase_to_bus: float
    v_from_bus_D: float 
    v_from_bus_Q: float  
    v_to_bus_D: float  
    v_to_bus_Q: float  
    i_br_D: float  
    i_br_Q: float  

@dataclass(slots=True)
class Series_rl_branch:
    idx: str
    from_bus: str
    to_bus: str
    sbase: float	
    vbase: float
    fbase: float
    r: float
    l: float
    name: str = field(default_factory=str)
    type: str = 'branch'
    pf: Optional[Power_flow_variables] = None
    emt_init_cond: Optional[EMT_initial_conditions] = None
    ssm: Optional[StateSpaceModel] = None

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.branches.loc[self.idx]     
        self.pf  = Power_flow_variables(vmag_from_bus = sol.from_bus_vmag.item(),
                                        vphase_from_bus = sol.from_bus_vphase.item(),
                                        vmag_to_bus = sol.to_bus_vmag.item(),
                                        vphase_to_bus = sol.to_bus_vphase.item())
        
        
    def _calculate_emt_initial_conditions(self):
        vmag_from_bus = self.pf.vmag_from_bus
        vphase_from_bus = self.pf.vphase_from_bus

        vmag_to_bus = self.pf.vmag_to_bus
        vphase_to_bus =  self.pf.vphase_to_bus

        v_from_bus_DQ = vmag_from_bus*np.exp(vphase_from_bus*np.pi/180*1j) 
        v_to_bus_DQ = vmag_to_bus*np.exp(vphase_to_bus*np.pi/180*1j) 

        i_br_DQ = (v_from_bus_DQ - v_to_bus_DQ)/(self.r + 1j*self.l)

        self.emt_init_cond = EMT_initial_conditions(vmag_from_bus = vmag_from_bus,
                                                    vphase_from_bus = vphase_from_bus,
                                                    vmag_to_bus = vmag_to_bus,
                                                    vphase_to_bus = vphase_to_bus,
                                                    v_from_bus_D = v_from_bus_DQ.real,
                                                    v_from_bus_Q = v_from_bus_DQ.imag,
                                                    v_to_bus_D = v_to_bus_DQ.real,
                                                    v_to_bus_Q = v_to_bus_DQ.imag,
                                                    i_br_D = i_br_DQ.real,
                                                    i_br_Q = i_br_DQ.imag)

    def _build_small_signal_model(self):

        rse = self.r
        lse = self.l
        wb = 2*np.pi*self.fbase
    
        # Define state-space matrices
        A = wb*np.array([[-rse/lse, 1],
                         [-1,      -rse/lse]])

        B = wb*np.array([[1/lse,  0,  -1/lse,  0],
                         [0,   1/lse,  0,  -1/lse]])

        C = np.eye(2)

        D = np.zeros((2,4))
        
        u = Variables(
            name=["v_from_bus_D", "v_from_bus_Q", "v_to_bus_D", "v_to_bus_D"],
            component=[self.idx]*4,
            v_type=["grid"]*4,
            init=[self.emt_init_cond.v_from_bus_D, self.emt_init_cond.v_from_bus_Q,
                  self.emt_init_cond.v_to_bus_D,   self.emt_init_cond.v_to_bus_Q]
        )
        
        x = Variables(
            name=["i_br_D", "i_br_Q"],
            component=[self.idx]*2,
            v_type=["grid"]*2,
            init=[self.emt_init_cond.i_br_D, self.emt_init_cond.i_br_Q]
        )
        

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=x, x=x)