# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
from typing import ClassVar
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
class Bus(Component):
    #id: int = field(default=-1, init=False)
    #name: str
    bus_type: str = None
    zone: str = None
    kron_removable_bus: bool = None
    base_power_MVA: float = None
    base_voltage_kV: float = None
    base_frequency_Hz: float = None
    latitude: float = None
    longitude: float = None
    max_flow_MW: float = None
    minimum_voltage_pu: float = None
    maximum_voltage_pu: float = None
    tags: ClassVar[list[str]] = []

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id

    def __repr__(self):
        return f"Bus(id={self.id}, bus='{self.name}')"
    
    def post_system_init(self, system):
        # We deduce max power flow based on the lines
        # the current bus is connected to.
        if self.max_flow_MW is not None:
            return  # Already defined
        
        self.max_flow_MW = 0.0
        connected_lines = {line for line in system.lines if (self.name in [line.from_bus, line.to_bus])}

        for line in connected_lines:
            # There is no constraint on max power flow on the line,
            # thus the bus should also inherit no constraint (and we can exit).
            if (line.cap_existing_power_MW is None):
                self.max_flow_MW = None
                return
            # Otherwise
            else:
                self.max_flow_MW += line.cap_existing_power_MW

@dataclass(slots=True, kw_only=True)
class Load(Component):
    #id: int = field(default=-1, init=False)
    bus: str
    timepoint: str
    load_MW: float
    name: str = 'unnamed_load'
    scenario: str = None
    load_MVAR: float = None

    def __repr__(self):
        return f"Load(id={self.id}, bus='{self.bus}', timepoint='{self.timepoint}')"

    

                                
                                  


