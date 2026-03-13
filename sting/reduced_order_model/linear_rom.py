# -------------
# Import python packages
# --------------
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

# -------------
# Import sting code
# -------------
from sting.utils.dynamical_systems import StateSpaceModel
from sting.system.component import Component

@dataclass
class Reducer(ABC):
    """An abstract base class for model reduction methods."""

    @abstractmethod
    def reduce(self, sys) -> StateSpaceModel:
        pass

@dataclass(slots=True, kw_only=True, eq=False)
class LinearROM(Component):
    ssm: StateSpaceModel = None # reduced order model
    full_order_model: StateSpaceModel = None
    reducer: Reducer = None
    

    W_c: np.ndarray = None # Make this an object with method and values
    W_o: np.ndarray = None
    
    T_l: np.ndarray = None # left projection matrix
    T_r: np.ndarray = None # right projection matrix


    def _construct_rom(self):
        self.reducer.reduce(sys=self)

