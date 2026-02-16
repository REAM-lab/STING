# ----------------
# Import python packages
# ----------------
from dataclasses import dataclass
from typing import NamedTuple

# ----------------
# Classes
# ----------------
class SystemComponent(NamedTuple):
    type_: str
    class_: str

@dataclass(slots=True, kw_only=True)
class Component:
    id: int = 0
    name: str
    type_: str = 'component'

