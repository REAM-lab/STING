from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class Bus:
    idx: int = field(default=-1, init=False)
    sbase: float
    vbase: float
    fbase: float
    v_min: float
    v_max: float
    p_load: float
    q_load: float
    name: str = field(default_factory=str)
    tags: ClassVar[list[str]] = []