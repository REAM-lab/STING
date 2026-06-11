
# -----------
# Import python packages
# -----------
import numpy as np
from dataclasses import field
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
    Settings for the unit commitment model.
    """
    generator_type_costs: str = "linear"
    load_shedding: bool = False
    single_storage_injection: bool = False
    line_capacity: bool = True
    power_flow: str = "dc"
    angle_difference_limits: bool = False
    inspect_coefficients: bool = True
    write_model_file: bool = False

class SolverSettings(NamedTuple):
    """
    Settings for the solver for the unit commitment model.
    """
    solver_name: str = "mosek_direct"
    tee: bool = True
    solver_options: dict = field(default_factory=dict)