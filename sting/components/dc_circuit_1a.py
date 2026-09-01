from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel

# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    """Initial conditions for EMT simulation."""
    v_dc: float
    i_dc: float
    i_load: float 

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class DCCircuit1A:
    """DC side circuit 1A model."""
    
    c_dc: float
    r_dc: float 
    wbase: float

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, p: float, v_dc: float):

        # DC-side initial conditions 
        i_dc = p / v_dc  
        i_load = -v_dc/self.r_dc - i_dc 
        self.emt_init = InitialConditionsEMT(
                v_dc = v_dc,
                i_dc = i_dc,               
                i_load = i_load
            )

    def get_small_signal_model(self, i_load: float, i_dc: float, v_dc: float) -> StateSpaceModel:
        """
        Returns the small-signal state-space model of the DC circuit 1A.

        Inputs:
        - i_load: load current
        - i_dc: ac-side current 

        State-space representation in tableau form:

                │   Δx  │   Δu
        ────────────────────────
        dΔx/dt  │   A   │   B 
        ────────────────────────
        Δy      │   C   │   D

        Define the state vector, input vector, and output vector are:
        Δx = [Δv_dc]
        Δu = [Δi_load, Δi_dc]
        Δy = [Δv_dc]

        Then, the state-space matrices are defined as follows:

                   │ Δv_dc             │ Δi_load        Δi_dc
        ─────────────────────────────────────────────────────────────────────────────────────────────
        dΔv_dc/dt  │ -ωb/(c_dc*r_dc)   │ -ωb/c_dc      -ωb/c_dc
        ─────────────────────────────────────────────────────────────────────────────────────────────
        v_dc       │ 1                 │  0             0

        
        """
        # Parameters
        wbase = self.wbase
        c_dc = self.c_dc
        r_dc = self.r_dc 

        model = StateSpaceModel(
            A = wbase * np.array([[-1/(c_dc*r_dc)]]),
            B = wbase * np.array([[-1/c_dc, -1/c_dc]]),
            C = np.array([[1]]),
            D = np.array([[0, 0]]),
            u = DynamicalVariables(name=['i_load', 'i_dc'], 
                                   init=[i_load, i_dc]),
            y = DynamicalVariables(name=['v_dc'],
                                   init=[v_dc]),
            x = DynamicalVariables(name = ['v_dc'],
                                   init = [v_dc])
            )

        return model

    def define_variables_emt_dc(self):

        x = DynamicalVariables(
            name = ['v_dc'],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.v_dc]
        )

        u = DynamicalVariables(
            name=["i_load", "i_dc"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.i_load, self.emt_init.i_dc]

        )

        y = DynamicalVariables(
            name=["v_dc"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.v_dc]
        )

        return [x, u, y]

    def get_derivatives_step_emt_dc(self, i_load: float, i_dc: float, v_dc: float):
        """
        Compute the derivatives of the state variables for the DC circuit 1A model.
        Differential equations:
        dv_dc/dt = (ωb/c_dc) * (-i_dc - i_load + v_dc/r_dc)

        Inputs:
        - i_load: Load current [pu]
        - i_dc: DC current [pu]

        Returns:
        - List of derivatives [dv_dc/dt]
        """
        wb = self.wbase
        c_dc = self.c_dc
        r_dc = self.r_dc 

        d_v_dc = (wb/c_dc) * (-i_dc - i_load - v_dc/r_dc)

        return [d_v_dc]