# -------------
# Import python packages
# --------------
from dataclasses import dataclass
from typing import NamedTuple
import logging

# -------------
# Import sting code
# --------------
from sting.system.component import Component
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables


# Set up logging
logger = logging.getLogger(__name__)

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True, kw_only=True)
class Load(Component):
    bus: str
    timepoint: str
    load_MW: float
    scenario: str = None
    load_MVAR: float = None
    base_power_MVA: float = None
    base_voltage_kV: float = None
    base_frequency_Hz: float = None
    bus_id: int = None
    ssm: StateSpaceModel = None
    variables_emt: VariablesEMT = None
    id_variables_emt: dict = None
    modeled_as_other_load_type: bool = False


    def __repr__(self):
        return f"Load(id={self.id}, bus='{self.bus}', timepoint='{self.timepoint}')"

    def post_system_init(self, system):
        self.bus_id = next((n for n in system.buses if n.name == self.bus)).id
        self.base_power_MVA =system.buses[self.bus_id].base_power_MVA
        self.base_frequency_Hz = system.buses[self.bus_id].base_frequency_Hz
        self.base_voltage_kV = system.buses[self.bus_id].base_voltage_kV


    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return hash((self.id, self.type_))
    
    def __eq__(self, value: Component):
        """Equality based on id attribute, which must be unique for each instance."""
        return self.id == value.id and self.type_ == value.type_