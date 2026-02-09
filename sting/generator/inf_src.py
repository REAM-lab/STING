# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar
import pyomo.environ as pyo
import polars as pl
import os
from collections import defaultdict
import logging

# -------------
# Import sting code
# --------------
from sting.utils.data_tools import pyovariable_to_df, timeit
from sting.generator.generator import Generator

logger = logging.getLogger(__name__)

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class InfiniteSource2(Generator):
    pass
