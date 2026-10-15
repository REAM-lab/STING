from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel

# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    """Initial conditions for EMT simulation."""

    i_load: float
    i_eq: float 
    v_eq: float 
    gamma_load: float
    v_dc: float 

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class DCLoad2B:
    """ 
    Models a DC load as connected via a DC-DC converter, with an output capacitor and equivalent load resistance. 
    The voltage controller is modeled, the inner current controller is assumed to be instantaneous.
    
    """

    Cload: float # output capacitance
    wbase: float 
    v_eq_ref: float # output capacitor voltage reference 
    kv_p: float # proportional gain of the voltage controller
    kv_i: float # integral gain of the voltage controller
    p_load: float = None # nominal power consumption of the load 
    r_load: float = None # equivalent load resistance

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, v_dc: float, p_load: float):
        
        self.r_load = (self.v_eq_ref**2) / p_load
        self.p_load = p_load
        
        v_eq = self.v_eq_ref
        i_eq = v_eq / self.r_load
        gamma_load = i_eq # integrator initial state
        
        i_load = i_eq*v_eq/v_dc # load drawn from DC bus 

        self.emt_init = InitialConditionsEMT(
                v_eq = v_eq,
                i_eq = i_eq,
                i_load = i_load,
                gamma_load = gamma_load,
                v_dc = v_dc,
            )

    def get_small_signal_model(self) -> StateSpaceModel:
        
        # states are v_eq, gamma_load
        # inputs are v_dc, p_load, v_eq_ref
        # output is i_load 
        
        # state equations are given by:
        # dv_eq/dt = (i_eq - v_eq/r_load)/(Cload/wbase)
        # dgamma_load/dt = kv_i*(v_eq_ref - v_eq)
        
        # output equation is: i_load = i_eq*v_eq/v_dc 
        # where i_eq = kp*(v_eq_ref - v_eq) + gamma_load
        # i_load = i_eq*v_eq/v_dc = (kp*(v_eq_ref - v_eq) + gamma_load)*v_eq/v_dc 
        # linearizing this equation around equilibrium values of v_eq, gamma_load, and v_dc denoted with subscript 0
        v_eq0 = self.emt_init.v_eq
        gamma_load0 = self.emt_init.gamma_load
        vdc0 = self.emt_init.v_dc
        i_eq0 = self.kv_p*(self.v_eq_ref - v_eq0) + gamma_load0
        r_load0 = self.r_load
        
        # linearized output equation: i_load = iload0 + (di_eq/dv_eq)*(v_eq - v_eq0) + (di_eq/dgamma_load)*(gamma_load - gamma_load0) + (di_load/dv_dc)*(v_dc - vdc0)
        # where di_eq/dv_eq = -kv_p, di_eq/dgamma_load = 1, di_load/dv_dc = -i_eq0*v_eq0/(vdc0**2)
        # and the linearized output equation can be written as:
        # i_load ≈ iload0 - kv_p*(v_eq - v_eq0)*v_eq0/vdc0 + (gamma_load - gamma_load0)*v_eq0/vdc0 - i_eq0*v_eq0/(vdc0**2)*(v_dc - vdc0)  
        
        k = self.wbase / self.Cload 

        # writing the A matrix for the small signal model
        load = StateSpaceModel(
            A = np.array([[-k*(self.kv_p + 1/r_load0), k],
                          [-self.kv_i,                     0]]),
            B = np.array([[0, -k*v_eq0/self.v_eq_ref**2, k*(self.kv_p + 2/r_load0)],
                          [0,  0,                        self.kv_i]]),
            C = np.array([[(i_eq0 - self.kv_p*v_eq0)/vdc0, v_eq0/vdc0]]),
            D = np.array([[-i_eq0*v_eq0/vdc0**2, 0, self.kv_p*v_eq0/vdc0]]),
            u = DynamicalVariables(name=['v_dc', 'pload', 'v_eq_ref'],
                                   init=[self.emt_init.v_dc, self.p_load, self.v_eq_ref]),
            x = DynamicalVariables(name=['v_eq', 'gamma_load'],
                                   init=[self.emt_init.v_eq, self.emt_init.i_eq]),
            y = DynamicalVariables(
                name = ['i_load'],
                init = [self.emt_init.i_load]
        )
        )
        
        return load

    def define_variables_emt_dc(self):

        x = DynamicalVariables(
            name = ["v_eq", "gamma_load"],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.v_eq, self.emt_init.gamma_load]
        )

        u = DynamicalVariables(
            name=["v_dc", "p_load", "v_eq_ref"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.v_dc, self.p_load, self.v_eq_ref]

        )

        y = DynamicalVariables(
            name=["i_load"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.i_load]
        )

        return [x, u, y]

    def get_derivatives_step_emt_dc(self, v_eq: float, v_eq_ref: float, i_eq: float, r_load: float):

        d_v_eq = (i_eq - v_eq/r_load)*(self.wbase/self.Cload)
        d_gamma_load = self.kv_i*(v_eq_ref - v_eq)

        return [d_v_eq, d_gamma_load]
    
    def get_algebraic_step_emt_dc(self, v_eq: float, v_eq_ref: float, gamma_load: float, v_dc: float, p_load: float):
        r_load = (v_eq_ref**2)/p_load 
        i_eq = self.kv_p*(v_eq_ref - v_eq) + gamma_load
        i_load = i_eq*v_eq/v_dc
        return [i_eq, i_load, r_load]