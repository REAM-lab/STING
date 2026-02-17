"""
This module implements an infinite source that incorporates:
- Stiff voltage source: a voltage source with constant frequency and constant voltage.
- Series RL branch: It is in series with the stiff voltage source.
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
class LinearSystem(Component):
    emt_init: InitialConditionsEMT = None
    ssm: StateSpaceModel = None


