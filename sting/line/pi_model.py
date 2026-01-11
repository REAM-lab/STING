from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(slots=True)
class LinePiModel:
    id: int = field(default=-1, init=False)
    name: str 
    from_bus: str
    to_bus: str
    r_pu: float
    x_pu: float
    g_pu: float
    b_pu: float
    expand_capacity: bool = False
    cap_existing_power_MW: float = None
    cost_fixed_power_USDperkW: float = 0.0
    angle_max_deg: float = 360
    angle_min_deg: float = -360
    sbase_VA: float = None
    vbase_V: float = None
    fbase_Hz: float = None
    decomposed: bool = field(default=False)
    tags: ClassVar[list[str]] = ["line"]
    from_bus_id: int = None
    to_bus_id: int = None

    def assign_indices(self, system):
        self.from_bus_id = next((n for n in system.bus if n.name == self.from_bus)).id
        self.to_bus_id = next((n for n in system.bus if n.name == self.to_bus)).id
    
    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id