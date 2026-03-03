
# -----------
# Import python packages
# -----------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
import logging

# -----------
# Import sting code
# -----------
from sting.system.core import System

# Set up logging
logger = logging.getLogger(__name__)

# -----------
# Sub-classes 
# -----------
class ModelSettings(NamedTuple):   
    """
    Settings for the capacity expansion model.
    """
    generator_type_costs: str = "linear"
    load_shedding: bool = False
    single_storage_injection: bool = False
    generation_capacity_expansion: bool = True
    storage_capacity_expansion: bool = True
    line_capacity_expansion: bool = True
    line_capacity: bool = True
    power_flow: str = "dc"
    bus_max_flow_expansion: bool = False
    bus_max_flow: bool = False
    angle_difference_limits: bool = False
    write_model_file: bool = False
    kron_equivalent_flow_constraints: bool = False

class SolverSettings(NamedTuple):
    """
    Settings for the solver for the capacity expansion model.
    """
    solver_name: str = "mosek_direct"
    tee: bool = True
    solver_options: dict = field(default_factory=dict)

class KronVariables(NamedTuple):
    original_system: System
    removable_buses: set[str] = None
    Y_original: np.ndarray = None
    Y_kron: np.ndarray = None
    B_qp: np.ndarray = None
    invB_qq: np.ndarray = None

