# -------------
# Import python packages
# --------------
import numpy as np
from dataclasses import dataclass, field
from typing import Literal
from control import gram, StateSpace
# -------------
# Import sting code
# -------------
from sting.utils.dynamical_systems import StateSpaceModel
from sting.system.component import Component
from sting.modules.model_order_reduction.core import Reducer

#@dataclass
@dataclass(slots=True)
class Gramian:
    """State-space model gramians factored in *upper* Cholesky W = X' X"""
    type: Literal["controllability", "observability"]

    state_space: StateSpace = None
    subsystem: np.ndarray = None
    lyapunov: np.ndarray = None
    # structured: Not implemented
    # riccati: Not implemented

    def __getitem__(self, key):
        assert (key in {"subsystem", "lyapunov"})
        W = getattr(self, key)

        if (W is None) and (key == "subsystem"):
            W = gram(self.state_space, self.type[0]+"f")
            self.subsystem = W
        
        return W

@dataclass(slots=True, kw_only=True, eq=False)
class LinearSubsystem(Component):
    
    full_order_model: StateSpaceModel
    reduced_order_model: StateSpaceModel = None

    using: Literal["full_order_model", "reduced_order_model"] = "full_order_model"
    reducer: Reducer = None
    
    W_c: Gramian = field(default_factory=lambda : Gramian(type="controllability"))
    W_o: Gramian = field(default_factory=lambda : Gramian(type="observability"))
    
    T_l: np.ndarray = None # left projection matrix
    T_r: np.ndarray = None # right projection matrix

    def __post_init__(self):
        sys = self.full_order_model.to_python_control()
        self.W_c.state_space = sys
        self.W_o.state_space = sys

    @property
    def ssm(self):
        return getattr(self, self.using)
    
    def set_using(self, to):
        self.using = to

    def _construct_rom(self):
        self.reducer.reduce(sys=self)

