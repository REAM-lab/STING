from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel

# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    """Initial conditions for EMT simulation."""

    i_L: float
    v_dc: float
    i_dc: float
    d: float
    i_load: float


# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class DCCircuit2A:
    """DC side circuit 2A model."""

    v_s: float
    l_dc: float
    c_dc: float
    wbase: float

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, p: float, v_dc: float, i_load: float):

        # DC-side initial conditions 
        i_dc = p / v_dc  
        d = (v_dc - self.v_s) / v_dc 
        i_L = (i_load + i_dc) / (1 - d)

        self.emt_init = InitialConditionsEMT(
                i_L = i_L,
                v_dc = v_dc,
                i_dc = i_dc,
                d = d,                
                i_load = i_load
            )

    def get_small_signal_model(self, d: float, i_L: float, v_dc: float) -> StateSpaceModel:
        """
        Returns the small-signal state-space model of the DC circuit 2A.

        Inputs:
        - d: Initial value of duty cycle 
        - i_L: Initial value of inductor current [pu]
        - v_dc: Initial value of DC bus voltage [pu]

        State-space representation in tableau form:

                │   Δx  │   Δu
        ────────────────────────
        dΔx/dt  │   A   │   B 
        ────────────────────────
        Δy      │   C   │   D

        Define the state vector, input vector, and output vector are:
        Δx = [Δi_L, Δv_dc]
        Δu = [Δv_s, Δd, Δi_dc, Δi_load]
        Δy = [Δi_L, Δv_dc]

        Then, the state-space matrices are defined as follows:

                   │ Δi_L            Δv_dc            │ Δv_s       Δd                Δi_dc        Δi_load
        ─────────────────────────────────────────────────────────────────────────────────────────────
        dΔi_L/dt   │ 0               ωb*(dₒ-1)/l_dc   │ ωb*1/l_dc  ωb*(v_dc)ₒ/l_dc   0            0
        dΔv_dc/dt  │ ωb*(1-dₒ)/c_dc  0                │ 0          -ωb*(i_L)ₒ/c_dc   -ωb*1/c_dc   -ωb*1/c_dc
        ─────────────────────────────────────────────────────────────────────────────────────────────
        i_L        │ 1               0                │ 0          0                 0            0
        v_dc       │ 0               1                │ 0          0                 0            0

        
        """
        # Parameters
        wbase = self.wbase
        l_dc = self.l_dc
        c_dc = self.c_dc

        model = StateSpaceModel(
            A = wbase * np.array([[0,         (d-1)/l_dc], 
                                [(1-d)/c_dc,         0]]),
            B = wbase * np.array([[1/l_dc, v_dc/l_dc,       0,       0],
                                  [0,      -i_L/c_dc, -1/c_dc, -1/c_dc]]),
            C = np.eye(2),
            D = np.zeros((2,4)),
            u = DynamicalVariables(name=['v_s', 'd', 'i_dc','i_load']),
            y = DynamicalVariables(name=['i_L','v_dc']),
            x = DynamicalVariables(
                name = ['i_L', 'v_dc'],
                init = [i_L, v_dc]
        )
        )

        return model

    def define_variables_emt_dc(self):

        x = DynamicalVariables(
            name = ['i_L', 'v_dc'],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.i_L, self.emt_init.v_dc]
        )

        u = DynamicalVariables(
            name=["v_s", "d", "i_dc", "i_load"],
            component=f"{self.__class__.__name__}",
            init=[self.v_s, self.emt_init.d, self.emt_init.i_dc, self.emt_init.i_load]

        )

        y = DynamicalVariables(
            name=["i_L", "v_dc"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.i_L, self.emt_init.v_dc]
        )

        return [x, u, y]

    def get_derivatives_step_emt_dc(self, i_L: float, v_dc: float, v_s: float, d: float, i_load: float, i_dc: float):
        """
        Compute the derivatives of the state variables for the DC circuit 2A model.
        Differential equations:
        di_L/dt = (ωb/l_dc) * (v_s - (1 - d) * v_dc)
        dv_dc/dt = (ωb/c_dc) * (-i_dc - i_load + (1 - d) * i_L)

        Inputs:
        - i_L: Inductor current [pu]
        - v_dc: DC bus voltage [pu]
        - v_s: Source voltage [pu]
        - d: Duty cycle (0 <= d <= 1)
        - i_load: Load current [pu]
        - i_dc: DC current [pu]

        Returns:
        - List of derivatives [di_L/dt, dv_dc/dt]
        """
        wb = self.wbase
        l_dc = self.l_dc
        c_dc = self.c_dc

        d_i_L = (wb/l_dc) * (v_s - (1 - d) * v_dc)
        d_v_dc = (wb/c_dc) * (-i_dc - i_load + (1 - d) * i_L)

        return [d_i_L, d_v_dc]