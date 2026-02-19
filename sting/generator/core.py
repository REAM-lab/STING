# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple
import pyomo.environ as pyo
import polars as pl
import os

import logging

# -------------
# Import sting code
# --------------
from sting.system.component import Component
from sting.modules.power_flow.utils import ACPowerFlowSolution
from sting.utils.runtime_tools import timeit

# Set up logging
logger = logging.getLogger(__name__)

# ----------------
# Sub-classes
# ----------------
class PowerFlowVariables(NamedTuple):
    p_bus: float
    q_bus: float
    vmag_bus: float
    vphase_bus: float

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True, kw_only=True)
class Generator(Component):
    bus: str 
    minimum_active_power_MW: float = field(default=None, kw_only=True)
    maximum_active_power_MW: float = field(default=None, kw_only=True)
    minimum_reactive_power_MVAR: float = field(default=None, kw_only=True)
    maximum_reactive_power_MVAR: float = field(default=None, kw_only=True)
    base_power_MVA: float = None
    base_voltage_kV: float = None
    base_frequency_Hz: float = None
    technology: str = None
    site: str = None
    cap_existing_power_MW: float = None
    cap_max_power_MW: float = None
    cost_fixed_power_USDperkW: float = None
    cost_variable_USDperMWh: float = None
    c0_USD: float = None
    c1_USDperMWh: float = None
    c2_USDperMWh2: float = None
    emission_rate_tonneCO2perMWh: float = None
    tags: ClassVar[list[str]] = ["generator"]
    bus_id: int = None
    expand_capacity: bool = None
    component_id: str = None
    forced_dispatch_MW: float = None
    power_flow_variables: PowerFlowVariables = None

    def post_system_init(self, system):
        self.bus_id = next((n for n in system.buses if n.name == self.bus)).id

        if self.cap_existing_power_MW is not None and self.cap_max_power_MW is not None:
            self.expand_capacity = False if self.cap_existing_power_MW >= self.cap_max_power_MW else True

    def load_ac_power_flow_solution(self, timepoint: str, pf_solution: ACPowerFlowSolution):
        self.power_flow_variables = PowerFlowVariables(
            p_bus=pf_solution.generator_active_dispatch[self.id, timepoint, self.type_]/self.base_power_MVA,
            q_bus=pf_solution.generator_reactive_dispatch[self.id, timepoint, self.type_]/self.base_power_MVA,
            vmag_bus=pf_solution.bus_voltage_magnitude[self.bus_id, timepoint],
            vphase_bus=pf_solution.bus_voltage_angle[self.bus_id, timepoint],
        )
        
    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return hash((self.id, self.type_))
    
    def __eq__(self, value: Component):
        """Equality based on id attribute, which must be unique for each instance."""
        return self.id == value.id and self.type_ == value.type_
    
    def __repr__(self):
        return f"Generator(id={self.id}, name='{self.name}', bus='{self.bus}')"

@dataclass(slots=True)
class CapacityFactor(Component):
    site: str
    scenario: str
    timepoint: str
    capacity_factor: float
    technology: str = None


    
    