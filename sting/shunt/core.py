# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple
import pyomo.environ as pyo
import polars as pl
import os
from collections import defaultdict
import logging

# -------------
# Import sting code
# --------------
from sting.system.component import Component
from sting.modules.power_flow.utils import ACPowerFlowSolution
from sting.utils.data_tools import pyovariable_to_df, timeit

logger = logging.getLogger(__name__)

# ----------------
# Sub-classes
# ----------------
class PowerFlowVariables(NamedTuple):
    vmag_bus: float
    vphase_bus: float

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True, kw_only=True)
class Shunt(Component):
    bus: str
    base_power_MVA: float
    base_voltage_kV: float
    base_frequency_Hz: float
    type: str = "shunt"
    tags: ClassVar[list[str]] = ["shunt"]
    pf: PowerFlowVariables = None

    def post_system_init(self, system):
        self.bus_id = next((n for n in system.buses if n.name == self.bus)).id

    def load_ac_power_flow_solution(self,  timepoint: str, pf_solution: ACPowerFlowSolution):
        self.pf = PowerFlowVariables(
            vmag_bus=pf_solution.bus_voltage_magnitude[self.bus_id, timepoint],
            vphase_bus=pf_solution.bus_voltage_angle[self.bus_id, timepoint],
        )
