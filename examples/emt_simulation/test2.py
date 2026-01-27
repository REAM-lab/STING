from dataclasses import dataclass, field, asdict, fields
from typing import Optional
import numpy as np
import polars as pl

# A regular class, as dataclasses don't inherently support properties 
# in a way that automatically maps to backing fields.
class DynamicalVariables:
    __slots__ = ('_name', '_component', '_type', '_init', '_value', '_time')
    
    def __init__(self, 
                 name: list[str], 
                 component: str = None, 
                 type: list[str] = None,
                 init: list[np.ndarray] = None, 
                 value: list[np.ndarray] = None, 
                 time: np.ndarray = None):
        
        self._name = np.atleast_1d(name)
        self._component = np.full(len(self._name), component if component is not None else '') 
        self._type = np.full(len(self._name), type if type is not None else '') 
        self._init =np.full(len(self._name), init if init is not None else np.nan) 
        self._value = np.full((len(self._name),1), np.nan) if value is None else np.atleast_2d(value)
        self._time = np.atleast_1d(time) if time is not None else np.atleast_1d(np.nan)
    
    def __post_init__(self):

        for attr in self.__slots__:
            if attr in ['_name', '_time']:
                continue
            self.check_shapes(getattr(self, attr))

    # Name property and setter
    # --------------------------
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, new_value):
        raise AttributeError("Cannot modify 'name' attribute directly.")
    
    
    # Component property and setter
    # ------------------------------
    @property
    def component(self):
        return self._component
    
    @component.setter
    def component(self, new_value):
        new_value = np.full(len(self._name), new_value)
        self._component = new_value
    
    # Type property and setter
    # --------------------------
    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, new_value):
        new_value = np.atleast_1d(new_value).astype(str)
        self.check_shapes(new_value)
        self._type = new_value

    # Init property and setter
    # --------------------------
    @property
    def init(self):
        return self._init
    
    @init.setter
    def init(self, new_value):
        new_value = np.atleast_1d(new_value).astype(float)
        self.check_shapes(new_value)
        self._init = new_value

    # Value property and setter
    # --------------------------
    @property 
    def value(self):
        return self._value
    
    @value.setter
    def value(self, new_value):
        #if isinstance(new_value, np.ndarray):
        new_value = np.atleast_2d(new_value)
        #else:
        #    new_value = np.vstack(new_value)
        self.check_shapes(new_value)
        self._value = new_value
    
    # Time property and setter
    # --------------------------
    @property
    def time(self):
        return self._time
    
    @time.setter
    def time(self, new_value):
        new_value = np.atleast_1d(new_value).astype(float)
        self._time = new_value

    # Other properties
    # --------------------------
    @property
    def n_grid(self):
        """
        Number of variables of type 'grid'
        """
        return sum(self.type == "grid")

    @property
    def n_device(self):
        """ 
        Number of variables of type 'device'
        """
        return sum(self.type == "device")
    
    # Utility methods
    # --------------------------
    
    def check_shapes(self, new_value):
        if len(new_value) != len(self._name):
            raise ValueError(f"Length of attribute does not match length of 'name' ({len(self._name)}).")

    def to_list(self):
        # Return unique a tuple uniquely identifying each variable
        return list(zip(self.component.tolist(), self.name.tolist()))
    
    def to_csv(self):
        fields = list(self.__slots__)
        fields.remove('_time')
        fields.remove('_value')
        d = {k.lstrip('_'): getattr(self, k) for k in fields}
        df = pl.DataFrame(d)
        df.write_csv("test.csv")
    
    def to_timeseries(self):
        d = {k : self._value[i] for i, k in enumerate(self._name)}
        df = pl.DataFrame(d)
        new_col = pl.Series("time", self._time)
        df = df.insert_column(0, new_col)
        return df

    def __len__(self):
        return len(self.name)
    
    def __add__(self, other):
        # Concatenate to variables arrays column-wise
        if not np.array_equal(self.time, other.time):
            raise ValueError("Cannot add DynamicalVariables with different time arrays.")
        return DynamicalVariables(
            name=np.concatenate([self.name, other.name]),
            component=np.concatenate([self.component, other.component]),
            type=np.concatenate([self.type, other.type]),
            init=np.concatenate([self.init, other.init]),
            value=np.concatenate([self.value, other.value]),
            time=self.time) 
        
    def __getitem__(self, idx):
        return DynamicalVariables(
            name=self.name[idx],
            component=self.component[idx],
            type=self.type[idx],
            init=self.init[idx],
            value=self.value[idx],
            time=self.time
        )
    

    def __repr__(self):
        return f"""DynamicalVariables: 
        - name={self._name},
        - component={self._component},
        - type={self._type},
        - init={self._init},
        - value=..., 
        - time=...."""

x = DynamicalVariables(
            name=["v_bus_a", "v_bus_b", "v_bus_c"],
            type=["device", "grid", "grid"],
            )

x.value = [[1,2,3], [4,5,6], [7,8,9]]

u = DynamicalVariables(
            name=["theta_bus_a", "theta_bus_b"],
            component=["gen_1"],
            init=[0.0, 0.1])

l = DynamicalVariables(
            name=["p_load", "q_load"],
            component=["load_1"],
            init=[1.0, 2.0])

l.component = "new_component"  # This will raise an AttributeError
l.value = [[5, 6], [3,4]] 
u.value = [[5, 6], [3,4]] 
l.time = np.array([0, 1])
u.time = np.array([0, 1])


r = u + l

#r.to_csv()

xx = r.to_timeseries()

x[x.type == "grid"]
print('ok')