# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class Person:
    name: str
    age: int

x = [Person(name="Alice", age=30), Person(name="Bob", age=25)]

map(lambda p: p.age == 30, x)