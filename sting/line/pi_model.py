from dataclasses import dataclass, field

@dataclass
class LinePiModel:
    idx: str
    from_bus: str
    to_bus: str
    sbase: float	
    vbase: float
    fbase: float
    r: float
    l: float
    g: float
    b: float
    name: str = field(default_factory=str)
    type: str = 'line_pi'
    tags: list =  field(default_factory=lambda : list) 