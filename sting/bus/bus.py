from dataclasses import dataclass, field
from typing import Optional
@dataclass
class Bus:
    idx: int
    v_min: float
    v_max: float
    p_load: float	
    q_load: float
    name: str = field(default_factory=str)
    type: str = 'bus'
    tags: Optional[list] = field(default_factory=lambda : ["buses"]) 