# ----------------------
# Import python packages
# ----------------------
from __future__ import annotations
import polars as pl
import numpy as np
from dataclasses import dataclass, field
import os
import pyomo.environ as pyo
import time
import logging
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output
import importlib
from typing import NamedTuple

from pyomo.environ import *
from pyomo.repn import generate_standard_repn
import math

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
import sting.system.selections as sl
import sting.bus.bus as bus
import sting.generator.generator as generator
import sting.generator.storage as storage

logger = logging.getLogger(__name__)

# -----------
# Sub-classes 
# -----------
class SolverSettings(NamedTuple):
    """
    Settings for the solver for the capacity expansion model.
    """
    solver_name: str = "mosek_direct"
    tee: bool = True
    solver_options: dict = field(default_factory=dict)

# -----------
# Main class
# -----------
@dataclass(slots=True)
class ACPowerFlow:
    """
    Class for AC power flow model.
    """
    system: System
    simplified_system: System = None

    def __post_init__(self):
        attrs = ["name", "bus", "bus_id", "minimum_active_power_MW", "maximum_active_power_MW", "minimum_reactive_power_MVAR", "maximum_reactive_power_MVAR"]
        generators = self.system.generators.to_table_pl(*attrs)

        self.simplified_system = System()

        for row in generators.iter_rows(named=True):
                component = generator.Generator(**row)
                self.simplified_system.add(component)

        print("ok")