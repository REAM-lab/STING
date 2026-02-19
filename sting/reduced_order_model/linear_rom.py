"""

"""
# -------------
# Import python packages
# --------------
import numpy as np
from scipy.linalg import block_diag
from dataclasses import dataclass
from typing import NamedTuple

# -------------
# Import sting code
# -------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.system.component import Component

# -------------
# Sub-classes
# -------------
class InitialConditionsEMT(NamedTuple):
    pass


@dataclass(slots=True, kw_only=True, eq=False)
class LinearROM(Component):
    ssm: StateSpaceModel = None


