from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
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
    tags: ClassVar[list[str]] = ["line"]
