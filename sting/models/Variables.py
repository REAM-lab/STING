import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class Variables:
    """
    A lightweight class used to hold data about the variables 
    of a dynamical system, such as inputs, outputs, and states.

    name: Name of the variable (e.g., i_d, v_q, etc.)
    component: Unique name of the component associated with each state.
    v_type: Variable type (i.e, 'device' or 'grid')
    init: Initial conditions
    """
    name: any
    component: any = None
    v_type: any = None
    init: any = None

    def __post_init__(self):
        # Convert fields to NumPy arrays if they aren't already
        self.name = np.asarray(self.name)
        
        # Infer number of variables
        n = len(self.name)

        # Convert provided fields or initialize defaults
        self.component = np.asarray(self.component) if self.component is not None else np.full(n, '', dtype=str)
        self.v_type   = np.asarray(self.v_type)     if self.v_type is not None    else np.full(n, 'grid', dtype=str)
        self.init     = np.asarray(self.init)       if self.init is not None      else np.full(n, np.nan, dtype=float)

        # Enforce consistent lengths
        lengths = {len(self.name), len(self.component), len(self.v_type), len(self.init)}
        if len(lengths) != 1:
            raise ValueError("All fields must have the same length.")
            
    @property
    def n_grid(self):
        return sum(self.v_type == 'grid')
    
    @property
    def n_device(self):
        return sum(self.v_type == 'device')
        

    def __add__(self, other):
        # Concatenate to variables arrays column-wise
        return Variables(
            name=np.concatenate([self.name, other.name]),
            component=np.concatenate([self.component, other.component]),
            v_type=np.concatenate([self.v_type, other.v_type]),
            init=np.concatenate([self.init, other.init]),
        )

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        # Return a sliced Variables object.
        if type(idx) == int:
            idx = np.array([idx])
        elif type(idx) != np.ndarray:
            idx = np.array(idx)
            
        return Variables(
            name=self.name[idx],
            component=self.component[idx],
            v_type=self.v_type[idx],
            init=self.init[idx])
    
    def to_list(self):
        # Return unique a tuple uniquely identifying each variable 
        return list(zip(self.component.tolist(), self.name.tolist()))