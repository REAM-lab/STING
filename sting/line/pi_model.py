from dataclasses import dataclass, field

@dataclass
class Line_no_series_compensation:
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
    type: str = 'line_ns'