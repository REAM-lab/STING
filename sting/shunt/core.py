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
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

logger = logging.getLogger(__name__)

# ----------------
# Sub-classes
# ----------------
class PowerFlowVariables(NamedTuple):
    vmag_bus: float
    vphase_bus: float

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True, kw_only=True)
class Shunt(Component):
    bus: str
    base_power_MVA: float
    base_voltage_kV: float
    base_frequency_Hz: float
    tags: ClassVar[list[str]] = ["shunt"]
    bus_id: int = None
    power_flow_variables: PowerFlowVariables = None
    ssm: StateSpaceModel = None
    variables_emt: VariablesEMT = None
    id_variables_emt: dict = None


    def post_system_init(self, system):
        self.bus_id = next((n for n in system.buses if n.name == self.bus)).id

    def load_ac_power_flow_solution(self,  timepoint: str, pf_solution: ACPowerFlowSolution):
        self.power_flow_variables = PowerFlowVariables(
            vmag_bus=pf_solution.bus_voltage_magnitude[self.bus_id, timepoint],
            vphase_bus=pf_solution.bus_voltage_angle[self.bus_id, timepoint],
        )
