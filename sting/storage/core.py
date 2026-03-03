# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, Optional
import pyomo.environ as pyo
import polars as pl
import logging

# -------------
# Import sting code
# --------------
from sting.system.component import Component

# Set up logging
logger = logging.getLogger(__name__)

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class Storage(Component):
    technology: str
    bus: str
    cap_existing_energy_MWh: float
    cap_existing_power_MW: float
    cap_max_power_MW: float
    cost_fixed_energy_USDperkWh: float
    cost_fixed_power_USDperkW: float
    cost_variable_USDperMWh: float
    duration_hr: float
    efficiency_charge: float
    efficiency_discharge: float
    c0_USD: float
    c1_USDperMWh: float
    c2_USDperMWh2: float
    expand_capacity: bool = True
    bus_id: int = None

    def post_system_init(self, system):
        self.expand_capacity = False if self.cap_existing_power_MW >= self.cap_max_power_MW else True
        self.bus_id = next((n for n in system.buses if n.name == self.bus)).id

    def __repr__(self):
        return f"Storage(id={self.id})"
    
    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id
