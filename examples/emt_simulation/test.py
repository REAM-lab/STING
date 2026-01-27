from dataclasses import dataclass, field, asdict, fields
from typing import Optional
import numpy as np
import polars as pl

@dataclass(slots=True)
class DynamicalVariables:
    name: list[str]
    component: Optional[str] = field(default=None)
    value: Optional[np.ndarray] = field(default=None)
    time: Optional[np.ndarray] = field(default=None)
    _initialized: bool = field(default=False)

    def __post_init__(self):
        
        self.name = np.asarray(self.name, dtype=str)
        self.name = np.atleast_1d(self.name)
      
        self.component = np.full(len(self.name), self.component)

        self.check_shapes()


        self.value =np.full(len(self.name), self.value) 
        self.time = np.atleast_1d(self.time) if self.time is not None else None

        object.__setattr__(self, '_initialized', True) 

        

    def check_shapes(self):

        for field in fields(self):
            attribute = getattr(self, field.name)
            if attribute is not None:
                if field.name in ["time", "_initialized"]:
                    continue
                if len(attribute) != len(self.name):
                    raise ValueError(f"Length of attribute '{field.name}' ({len(attribute)}) does not match length of 'name' ({len(self.name)}).")
                
    def on_attribute_modified(self, name: str, old_value, new_value):
        self.check_shapes()

    def __setattr__(self, name, value):
        """
        Intercepts all attribute assignments to the dataclass instance.
        """
        # Step 1: Check if the instance is already initialized.
        # During __init__ and __post_init__, we want to bypass the modification logic.
        if hasattr(self, '_initialized') and getattr(self, '_initialized'):
            # Step 2: Get the old value if the attribute already exists.
            # Use getattr with a default of None to handle new attributes being set.
            old_value = getattr(self, name)

            # Step 3: Set the attribute using the base `object`'s setter.
            # This is CRUCIAL to prevent infinite recursion.
            object.__setattr__(self, name, value)

            # Step 4: Check if the value actually changed.
            # Note: For mutable objects (lists, dicts), `!=` checks reference equality.
            # For deeper comparison, you'd need a custom equality check (e.g., `old_value != value` might not be enough).
            if not np.array_equal(old_value, value):
                self.on_attribute_modified(name, old_value, value)
        else:
            # During initialization, just set the attribute normally.
            object.__setattr__(self, name, value)
    

    def __len__(self):
        return len(self.name)
    
    def __add__(self, other):
        # Concatenate to variables arrays column-wise
        if not np.array_equal(self.time, other.time):
            raise ValueError("Cannot add DynamicalVariables with different time arrays.")
        return DynamicalVariables(
            name=np.concatenate([self.name, other.name]),
            component=np.concatenate([self.component, other.component]),
            time=self.time) 
        

    def __getitem__(self, idx):
        return DynamicalVariables(
            name=self.name[idx],
            component=self.component[idx],
            time=self.time
        )
    
    def to_csv(self):
        d = asdict(self)
        d = {k: v.astype(str) if v.dtype == 'object' else v for k, v in d.items()}
        df = pl.DataFrame(d)
        df.write_csv("test.csv")

        

    #@property
    #def component(self) -> int:
    #    return self._component
    
    #@component.setter
    #def component(self, value: str):
    #    self._component = np.asarray([value] * len(self.name), dtype=str)

u = DynamicalVariables(
            name=["theta_bus_a"],
            component=["gen_1"])

l = DynamicalVariables(
            name=["p_load", "q_load"],
            component=["load_1"])

ul = u + l 
tps = [0, 1, 2, 3, 4, 5]
x = DynamicalVariables(
            name=["v_bus_a", "v_bus_b", "v_bus_c"],
            time= tps )

w = DynamicalVariables(
            name=["p_gen", "q_gen"],
            time= tps )
#x.component = np.full(len(x), "bus_1")

y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component="bus_1",
            time= tps )

r = x + y
r2 = x + w

p = DynamicalVariables(
            name=["var1", "var2"],
            time= tps )

p.value = np.array([10])
#r.to_csv()
print('ok')