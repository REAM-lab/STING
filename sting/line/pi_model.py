from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(slots=True)
class LinePiModel:
    from_bus: str
    to_bus: str
    sbase: float
    vbase: float
    fbase: float
    r: float
    l: float
    g: float
    b: float
    idx: int = field(default=-1, init=False)
    name: str = field(default_factory=str)
    decomposed: bool = field(default=False)
    tags: ClassVar[list[str]] = ["line"]

@dataclass(slots=True)
class Line:
    idx: int = field(default=-1, init=False)
    line: str
    bus_from: str
    bus_to: str
    r_pu: float
    x_pu: float
    g_pu: float
    b_pu: float
    rating_MVA: float
    transformer: bool
    angle_max_deg: float 
    angle_min_deg: float 
    bus_from_idx: int = field(default=None, init=False)
    bus_to_idx: int = field(default=None, init=False)

    def assign_indices(self, buses: list):
        self.bus_from_idx = next((n for n in buses if n.bus == self.bus_from)).idx
        self.bus_to_idx = next((n for n in buses if n.bus == self.bus_to)).idx

    def __hash__(self):
        return hash(self.idx)