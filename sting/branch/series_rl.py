from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import numpy as np
from sting.utils.linear_systems_tools import State_space_model

class Power_flow_variables(NamedTuple):
    vmag_from_bus: float = 0
    vphase_from_bus: float = 0
    vmag_to_bus: float = 0
    vphase_to_bus: float = 0

class EMT_initial_conditions(NamedTuple):
    vmag_from_bus: float = 0
    vphase_from_bus: float = 0
    x: np.ndarray = 0
    y: np.ndarray = 0

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
    pf: Power_flow_variables = field(default_factory=Power_flow_variables)
    emt_init_cond: EMT_initial_conditions = field(default_factory=EMT_initial_conditions)
    ssm: Optional[State_space_model] = None

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
        i_br_D, i_br_Q = np.real(i_br_DQ), np.imag(i_br_DQ)

        x = np.array([ i_br_D, i_br_Q])

        y = x

        local_vars = locals()
        data = {field: local_vars[field] for field in EMT_initial_conditions._fields}

        self.emt_init_cond = EMT_initial_conditions(**data)

    def _build_small_signal_model(self):

        rse = self.r
        lse = self.l
        wb = 2*np.pi*self.fbase
    
        A = wb*np.array([[-rse/lse, 1],
                             [-1,      -rse/lse]])

        B = wb*np.array([[1/lse,  0,  -1/lse,  0],
                             [0,   1/lse,  0,  -1/lse]])

        C = np.eye(2)

        D = np.zeros((2,4))

        grid_side_inputs = ["v1_d", "v1_q", "v2_d", "v2_q"]
        states = ["i_d", "i_q"]
        outputs = ["i_d", "i_q"]

        self.ssm = State_space_model(A = A, B = B, C = C, D = D, states=states, grid_side_inputs=grid_side_inputs, outputs=outputs)