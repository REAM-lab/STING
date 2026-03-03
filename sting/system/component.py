# ----------------
# Import python packages
# ----------------
from dataclasses import dataclass
from typing import NamedTuple

# ----------------
# Classes
# ----------------
class SystemComponent(NamedTuple):
    """Store a system component information."""
    type_: str
    class_: str

@dataclass(slots=True, kw_only=True)
class Component:
    """Defines a component in the system. It is the parent class of all components."""
    id: int = 0
    name: str = None
    type_: str = None
    zone: str = None

