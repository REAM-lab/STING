# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
from scipy.linalg import inv
import copy

# ------------------
# Import sting code
# ------------------
from sting.components.inner_current_controller_2a import InitialConditionsEMT
from sting.generator.core import Generator
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables, QuadraticBilinearModel
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.transformations import dq02abc, abc2dq0

from sting.utils.transformations import R_DQ2dq, R_dq2DQ, d_DQ2dq_dangle, d_dq2DQ_dangle

# ------------------
# Import sting code
# ------------------
class InitialConditionsEMT(NamedTuple):
    angle: float
    i_d: float
    i_q: float
    i_0: float
    i_fd: float
    i_1d: float
    i_1q: float
    i_2q: float
    v_fd: float
    v_d: float
    v_q: float
    v_0: float
    v_a: float
    v_b: float
    v_c: float
    i_D: float
    i_Q: float


@dataclass(slots=True, kw_only=True, eq=False)
class SynchronousMachine7A:
    x_d_pu: float 
    x_q_pu: float 
    x_ad_pu: float
    x_aq_pu: float
    x_fd_pu: float
    x_1d_pu: float
    x_1q_pu: float
    r_fd_pu: float
    r_1d_pu: float
    r_1q_pu: float
    w_base: float

    