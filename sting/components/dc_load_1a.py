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

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class DCLoad1A:

    Tload: float

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, i_load: float):

        self.emt_init = InitialConditionsEMT(
                i_load = i_load
            )

    def get_small_signal_model(self, i_load: float) -> StateSpaceModel:

        # Parameters
        Tload = self.Tload

        # Load control 
        load = StateSpaceModel(
            A = np.array([[-1/Tload]]),
            B = np.array([[1/Tload]]),
            C = np.array([[1]]),
            D = np.array([[0]]),
            u = DynamicalVariables(name=['i_load_ref'],
                                   init=[i_load]),
            y = DynamicalVariables(name=['i_load'],
                                   init=[i_load]),
            x = DynamicalVariables(
                name = ['i_load'],
                init = [i_load]
        )
        )
        return load

    def define_variables_emt_dc(self):

        x = DynamicalVariables(
            name = ["i_load"],
            component = f"{self.__class__.__name__}",
            init = [self.emt_init.i_load]
        )

        u = DynamicalVariables(
            name=["i_load_ref"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.i_load]

        )

        y = DynamicalVariables(
            name=["i_load"],
            component=f"{self.__class__.__name__}",
            init=[self.emt_init.i_load]
        )

        return [x, u, y]

    def get_derivatives_step_emt_dc(self, i_load: float, i_load_ref: float):
        
        # Parameters
        Tload = self.Tload

        # Load control 
        d_i_load = 1/Tload * (i_load_ref - i_load)

        return [d_i_load]