import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
from sting.utils.linear_systems_tools import State_space_model

class Power_flow_variables(NamedTuple):
    vmag_bus: float = 0
    vphase_bus: float = 0

class EMT_initial_conditions(NamedTuple):
    vmag_bus: float = 0
    vphase_bus: float = 0
    v_bus_DQ: float = 0
    i_bus_DQ: complex = 0
    x: np.ndarray = 0
    y: np.ndarray = 0

@dataclass
class Parallel_rc_shunt:
    idx: str
    bus_idx: str
    sbase: float	
    vbase: float
    fbase: float
    r: float
    c: float
    pf: Power_flow_variables = field(default_factory=Power_flow_variables)
    emt_init_cond: EMT_initial_conditions = field(default_factory=EMT_initial_conditions)
    ssm: Optional[State_space_model] = None
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
        v_bus_D, vbus_Q =  np.real(v_bus_DQ), np.imag(v_bus_DQ) 
        
        i_bus_DQ =  v_bus_DQ*self.g +  v_bus_DQ*(1j*self.b)

        x = np.array([ v_bus_D, vbus_Q])

        y = np.array([ v_bus_D, vbus_Q])

        local_vars = locals()
        data = {field: local_vars[field] for field in EMT_initial_conditions._fields}

        self.emt_init_cond = EMT_initial_conditions(**data)

    def _build_small_signal_model(self):
        g = self.g
        b = self.b
        wb = 2*np.pi*self.fbase
    
        A = wb*np.array([[-g/b, 1],
                         [-1, -g/b]])

        B = wb*np.array([[1/b, 0],
                         [0,    1/b]])

        C = np.eye(2)

        D = np.zeros((2,2))

        grid_side_inputs = ["i_d", "i_q"]
        states = ["v_d", "v_q"]
        outputs = ["v_d", "v_q"]

        self.ssm = State_space_model(A = A, B = B, C = C, D= D, grid_side_inputs=grid_side_inputs, states=states, outputs=outputs)