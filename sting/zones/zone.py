"""
This module implements a zone level small-signal / EMT model consisting of multiple other dynamic components.
"""

import numpy as np
from scipy.linalg import block_diag
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, ClassVar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.utils.transformations import dq02abc, abc2dq0



class InitialConditionsEMT(NamedTuple):
    v_bus_D: float
    v_bus_Q: float
    v_int_d: float
    v_int_q: float
    i_bus_d: float
    i_bus_q: float
    i_bus_D: float
    i_bus_Q: float
    angle_ref: float


@dataclass(slots=True)
class Zone:
    id: int = field(default=0, init=False)
    name: str 
    components: list
    connections: np.array
    minimal_outputs: bool = True

    #emt_init: Optional[InitialConditionsEMT] = None
    ssm: Optional[StateSpaceModel] = None
    type: str = "zone"
    tags: ClassVar[list[str]] = []
    #variables_emt: Optional[VariablesEMT] = None
    #id_variables_emt: Optional[dict] = None



    def __post_init__(self):
        """
        minimal_outputs: If True zonal model will contain only (the minimal set of) port facing outputs.
            If False, zone output will contain the outputs of every component.
        """

        pass

        # self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)

    def _calculate_emt_initial_conditions(self):
        pass
