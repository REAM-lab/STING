from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel

# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    """Initial conditions for EMT simulation."""

    x_1: float
    x_2: float
    i_L_f: float
    v_dc_f: float
    i_dc_f: float
    i_load_f: float
    d: float

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class DCController6A:

    TiL: float
    Tvdc: float
    Tidc: float
    Ti_load: float
    kp_vdc: float
    ki_vdc: float
    kp_iL: float
    ki_iL: float
    kff_idc: float
    kff_iload: float

    emt_init: InitialConditionsEMT = field(init=False)



    def get_steady_state(self, i_L: float, i_dc: float, i_load: float, d: float, v_dc: float):

        x_1 = i_L - self.kff_idc*i_dc - self.kff_iload*i_load 
        x_2 = d - self.kp_iL*(x_1 - i_L + self.kff_idc*i_dc + self.kff_iload*i_load) 

        self.emt_init = InitialConditionsEMT(
                x_1 = x_1,
                x_2 = x_2,
                i_L_f = i_L,
                v_dc_f = v_dc,
                i_dc_f = i_dc,
                i_load_f = i_load,
                d = d
            )

    def get_small_signal_model(self, i_dc: float, i_L: float, v_dc: float, i_load: float, x_1: float, x_2: float, d: float) -> StateSpaceModel:

        # Parameters
        TiL = self.TiL
        Tvdc = self.Tvdc
        Tidc = self.Tidc
        Ti_load = self.Ti_load
        kp_vdc = self.kp_vdc
        ki_vdc = self.ki_vdc
        kp_iL = self.kp_iL
        ki_iL = self.ki_iL
        kff_idc = self.kff_idc
        kff_iload = self.kff_iload

        dc_dc_controller = StateSpaceModel(
                A = np.array([
                                [-1/TiL, 0, 0, 0, 0, 0],
                                [0,-1/Tvdc, 0, 0, 0, 0],
                                [0, 0, -1/Tidc, 0, 0, 0],
                                [0, 0, 0, -1/Ti_load, 0, 0],
                                [0, -ki_vdc, 0, 0, 0, 0],
                                [-ki_iL, -kp_vdc*ki_iL, kff_idc*ki_iL, kff_iload*ki_iL, ki_iL, 0]]),
                B = np.array([
                                [1/TiL, 0, 0, 0, 0],
                                [0, 1/Tvdc, 0, 0, 0],
                                [0, 0, 1/Tidc, 0, 0],
                                [0, 0, 0, 1/Ti_load, 0],
                                [0, 0, 0, 0, ki_vdc],
                                [0, 0, 0, 0, ki_iL*kp_vdc]]),
                
                C = np.array([[-kp_iL, -kp_iL*kp_vdc, kp_iL*kff_idc, kp_iL*kff_iload, kp_iL, 1]]),
                D = np.array([[0, 0, 0, 0, kp_iL*kp_vdc]]),
                u = DynamicalVariables(name=['i_L', 'v_dc', 'i_dc', 'i_load', 'v_dc_ref'],
                                       init = [i_L, v_dc, i_dc, i_load, v_dc]),
                y = DynamicalVariables(name=['d'],
                                       init = [d]),
                x = DynamicalVariables(
                    name = ['i_L_f', 'v_dc_f', 'i_dc_f', 'i_load_f','x_1', 'x_2'],
                    init = [i_L, v_dc, i_dc, i_load, x_1, x_2]
                )
            )

        return dc_dc_controller

    def define_variables_emt_dc(self):

        x = DynamicalVariables(
            name = ["i_L_f", "v_dc_f", "i_dc_f", "i_load_f","x_1", "x_2"],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.i_L_f, self.emt_init.v_dc_f, self.emt_init.i_dc_f, self.emt_init.i_load_f, self.emt_init.x_1, self.emt_init.x_2]
        )

        u = DynamicalVariables(
            name=["i_L", "v_dc", "i_dc", "i_load", "v_dc_ref"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.i_L_f, self.emt_init.v_dc_f, self.emt_init.i_dc_f, self.emt_init.i_load_f, self.emt_init.v_dc_f]
        )

        y = DynamicalVariables(
            name=["d"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.d]
        )

        return [x, u, y]

    def get_derivatives_step_emt_dc(self, i_L_f: float, v_dc_f: float, i_dc_f: float, i_load_f: float, x_1: float, x_2: float,
                                          i_L: float, v_dc: float, i_dc: float, i_load: float, v_dc_ref: float):
        # Parameters
        TiL = self.TiL
        Tvdc = self.Tvdc
        Tidc = self.Tidc
        Ti_load = self.Ti_load
        kp_vdc = self.kp_vdc
        ki_vdc = self.ki_vdc
        kp_iL = self.kp_iL
        ki_iL = self.ki_iL
        kff_idc = self.kff_idc
        kff_iload = self.kff_iload

        # Derivatives
        d_i_L_f = (1/TiL)*(i_L - i_L_f)
        d_v_dc_f = (1/Tvdc)*(v_dc - v_dc_f)
        d_i_dc_f = (1/Tidc)*(i_dc - i_dc_f)
        d_i_load_f = (1/Ti_load)*(i_load - i_load_f)
        d_x_1 = ki_vdc*(v_dc_ref - v_dc_f)
        d_x_2 = ki_iL*(kp_vdc*(v_dc_ref - v_dc_f) + x_1 - i_L_f + kff_idc*i_dc_f + kff_iload*i_load_f)
        
        return [d_i_L_f, d_v_dc_f, d_i_dc_f, d_i_load_f, d_x_1, d_x_2]

    def get_algebraics_step_emt_dc(self, i_L_f: float, v_dc_f: float, i_dc_f: float, i_load_f: float, x_1: float, x_2: float, v_dc_ref: float):
        # Parameters
        kp_vdc = self.kp_vdc
        kp_iL = self.kp_iL
        kff_idc = self.kff_idc
        kff_iload = self.kff_iload

        # Algebraics
        duty_cycle = kp_iL*(kp_vdc*(v_dc_ref - v_dc_f) + x_1 - i_L_f + kff_idc*i_dc_f + kff_iload*i_load_f) + x_2
        
        return duty_cycle
