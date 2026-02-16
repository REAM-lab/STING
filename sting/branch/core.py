# -------------
# Import python packages
# --------------
from dataclasses import dataclass
from typing import ClassVar, NamedTuple
import logging

# -------------
# Import sting code
# --------------
from sting.system.component import Component
from sting.modules.power_flow.utils import ACPowerFlowSolution

# Set up logging
logger = logging.getLogger(__name__)

# ----------------
# Sub-classes
# ----------------
class PowerFlowVariables(NamedTuple):
    vmag_from_bus: float
    vphase_from_bus: float
    vmag_to_bus: float
    vphase_to_bus: float

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True, kw_only=True)
class Branch(Component):
    from_bus: str
    to_bus: str
    base_power_MVA: float
    base_voltage_kV: float
    base_frequency_Hz: float
    tags: ClassVar[list[str]] = ["branch"]
    pf: PowerFlowVariables = None

    def post_system_init(self, system):
        self.from_bus_id = next((n for n in system.buses if n.name == self.from_bus)).id
        self.to_bus_id = next((n for n in system.buses if n.name == self.to_bus)).id

    def load_ac_power_flow_solution(self, timepoint: str, pf_solution: ACPowerFlowSolution):
        self.pf = PowerFlowVariables(
            vmag_from_bus=pf_solution.bus_voltage_magnitude[self.from_bus_id, timepoint],
            vphase_from_bus=pf_solution.bus_voltage_angle[self.from_bus_id, timepoint],
            vmag_to_bus=pf_solution.bus_voltage_magnitude[self.to_bus_id, timepoint],
            vphase_to_bus=pf_solution.bus_voltage_angle[self.to_bus_id, timepoint],
        )
