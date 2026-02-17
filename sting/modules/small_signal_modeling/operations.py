# -----------------------
# Import python packages
# -----------------------
import logging
from dataclasses import dataclass
from itertools import groupby
import copy

import numpy as np
from scipy.linalg import block_diag
# -----------------------
# Import sting code
# -----------------------
from sting.generator.linear_system import LinearSystem
from sting.modules.small_signal_modeling.core import SmallSignalModel, ComponentSSM
from sting.utils.dynamical_systems import StateSpaceModel
from sting.system.core import System
from sting.utils.data_tools import mat2cell, cell2mat

# Set up logger
logger = logging.getLogger(__name__)


