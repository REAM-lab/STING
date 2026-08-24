import numpy as np
from typing import NamedTuple
from sting.utils.dynamical_systems import DynamicalVariables


# -----------
# Sub-classes
# -----------
class VariablesSSM(NamedTuple):
    """
    All variables in the system for small-signal modeling.
    """
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables

class ComponentSSM(NamedTuple):
    """
    A component of the system that participates in small-signal modeling.

    #### Attributes:
    - type: `str`
            inf_src, se_rl, pa_rc, ... etc. 
    - idx: `int`
            Index of the component in its corresponding list in the system.
    """
    type: str
    id: int

class ConnectionMatrices(NamedTuple):
    """
    Component connection matrices
    Using a NamedTuple to avoid accessing each element by it's index in a list
    """
    F: np.ndarray
    G: np.ndarray
    H: np.ndarray
    L: np.ndarray