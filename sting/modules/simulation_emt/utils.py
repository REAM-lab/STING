# ----------------------
# Import python packages
# ----------------------
import numpy as np
from typing import NamedTuple

# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import DynamicalVariables

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables