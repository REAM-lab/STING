from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel

# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    """Initial conditions for EMT simulation."""
    x_oc: float 
    i_d: float 
    v_dc: float 

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class DCVoltageController1A:

    kp_vdc_pu: float
    ki_vdc_puHz: float
    v_dc_ref: float 

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, v_dc: float, i_d: float):
        
        self.emt_init = InitialConditionsEMT(
                x_oc = i_d, 
                i_d = i_d,
                v_dc = v_dc,
            )

    def get_small_signal_model(self, v_dc: float, i_d: float) -> StateSpaceModel:

        # Parameters
        kp_vdc = self.kp_vdc_pu
        ki_vdc = self.ki_vdc_puHz 
        
        controller = StateSpaceModel(
            A = np.array([[0]]),
            B = np.array([[-ki_vdc, ki_vdc]]),
            C = np.array([[1]]),
            D = np.array([[-kp_vdc, kp_vdc]]),
            u = DynamicalVariables(name=['v_dc_ref', 'v_dc'],
                                   init=[v_dc, v_dc]),
            y = DynamicalVariables(name=['i_d_ref'],
                                   init=[i_d]),
            x = DynamicalVariables(
                name = ['x_oc'],
                init = [v_dc]
        )
        )
        return controller

    def define_variables_emt_dc(self):

        x = DynamicalVariables(
            name = ["x_oc"],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.x_oc]
        )

        u = DynamicalVariables(
            name=["v_dc_ref", "v_dc"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.v_dc, self.emt_init.v_dc]

        )

        y = DynamicalVariables(
            name=["i_d_ref"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.i_d]
        )

        return [x, u, y]

    def get_derivatives_step_emt_dc(self, v_dc_ref: float, v_dc: float):
        
        d_x_oc = self.ki_vdc_puHz*(-v_dc_ref + v_dc)

        return [d_x_oc]
    
    def get_algebraics_step_emt_dc(self, x_oc: float, v_dc_ref: float, v_dc: float):
        
        i_d_ref = self.kp_vdc_pu*(-v_dc_ref + v_dc) + x_oc 
        
        return i_d_ref
        