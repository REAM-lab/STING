# -------------
# Import python packages
# --------------
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

# -------------
# Import sting code
# -------------
from sting.utils.dynamical_systems import StateSpaceModel
from sting.system.component import Component
from sting.modules.model_order_reduction.utils import Grammian
from sting.modules.model_order_reduction.core import Reducer

@dataclass(slots=True, kw_only=True, eq=False)
class LinearROM(Component):
    reduced_order_model: StateSpaceModel = None
    full_order_model: StateSpaceModel = None

    using: Literal["full_order_model", "reduced_order_model"] = "full_order_model"
    reducer: Reducer = None
    
    W_c: Grammian = field(default_factory=lambda _: Grammian(type="controllability"))
    W_o: Grammian = field(default_factory=lambda _: Grammian(type="observability"))
    
    T_l: np.ndarray = None # left projection matrix
    T_r: np.ndarray = None # right projection matrix

    @property
    def ssm(self):
        return getattr(self, self.using)
    
    def set_using(self, to):
        self.using = to

    def _construct_rom(self):
        self.reducer.reduce(sys=self)

