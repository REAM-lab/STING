# ----------------------
# Import python packages
# ----------------------
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from sting.utils.dynamical_systems import StateSpaceModel

@dataclass
class Reducer(ABC):
    """An abstract base class for model reduction methods."""
    r: int
    system_operations: list = field(default_factory=list)

    @abstractmethod
    def reduce(self, sys) -> StateSpaceModel:
        pass

