# Import standard python packages and third-party packages
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import pandas as pd

# Import sting packages

from sting.models.StateSpaceModel import StateSpaceModel
from sting.models.Variables import Variables

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
    idx: str
    bus_idx: str
    sbase: float	
    vbase: float
    fbase: float
    r: float
    c: float
    pf: Optional[Power_flow_variables] = None
    emt_init_cond: Optional[EMT_initial_conditions] = None
    ssm: Optional[StateSpaceModel] = None
    name: str = field(default_factory=str)
    type: str = 'shunt'

    @property
    def g(self):
        return 1/self.r
    
    @property
    def b(self):
        return 1/self.c

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.shunts.loc[self.idx]
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

        u = Variables(
            name=["i_bus_D", "i_bus_Q"],
            component=[self.idx]*2,
            v_type=["grid"]*2,
            init=[self.emt_init_cond.i_bus_D, self.emt_init_cond.i_bus_Q]
        )
        
        x = Variables(
            name=["v_bus_D", "v_bus_Q"],
            component=[self.idx]*2,
            v_type=["grid"]*2,
            init=[self.emt_init_cond.v_bus_D, self.emt_init_cond.v_bus_Q]
        )

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=x, x=x)
        
        
def combine_shunts(system):

    print("> Reduce shunts to have one shunt per bus:")
   
    shs = system.components.pa_rc

    bus_idx = [s.bus_idx for s in shs]
    g = [s.g for s in shs]
    b = [s.b for s in shs]

    shunt_df = pd.DataFrame({'bus_idx': bus_idx, 'g': g, 'b': b})    
    shunt_df = shunt_df.pivot_table(index='bus_idx', values=['g', 'b'], aggfunc ='sum')
    shunt_df['r'] = 1/shunt_df['g']
    shunt_df['c'] = 1/shunt_df['b']
    shunt_df['idx'] = ['shred' + str(i) for i in range(len(shunt_df))] 
    shunt_df.reset_index(inplace=True)

    # Create new list of components "parallel rc shunts"
    system.componets.pa_rc = [] 

    # Add each effective/combined parallel RC shunt to the pa_rc components
    for _, row in shunt_df.iterrows(): 
        shunt = Parallel_rc_shunt(**row.to_dict())
        system.components.pa_rc.append(shunt) 
 
    print("\t- New list of parallel RC components created ... ok\n")