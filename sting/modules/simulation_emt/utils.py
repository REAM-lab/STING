# ----------------------
# Import python packages
# ----------------------
import numpy as np
from typing import NamedTuple
import inspect

# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import DynamicalVariables

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables

   
def modify_user_functions(func: callable):

    args = []

    parameters = inspect.signature(func).parameters

    if "x" in parameters:
        new_func = lambda t, x, id: func(t, x=x, id=id)
    else:
        new_func = lambda t, x, id: func(t)

    return new_func