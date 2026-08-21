from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel

# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    """Initial conditions for EMT simulation."""

    x1: float
    x2: float

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class DCsideController6A:

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


    def get_steady_state(self, i_L: float, i_dc: float, i_load: float, d: float, v_dc_f: float):

        x1 = i_L - self.kff_idc*i_dc - self.kff_iload*i_load 
        x2 = d - self.kp_iL*(x1 - i_L + self.kff_idc*i_dc + self.kff_iload*i_load) 

        self.emt_init = InitialConditionsEMT(
                x1 = x1,
                x2 = x2,
            )

    def get_small_signal_model(self, i_dc: float, i_L: float, v_dc: float, i_load: float, x1: float, x2: float, d: float) -> StateSpaceModel:

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
                    init = [i_L, v_dc, i_dc, i_load, x1, x2]
                )
            )

        return dc_dc_controller