import numpy as np
import os
import pandas as pd

from scipy.linalg import eigvals
from scipy.linalg import block_diag 
from dataclasses import dataclass
from sting.utils.data_tools import matrix_to_csv


@dataclass(slots=True)
class DynamicalVariables:
    """
    A lightweight class used to hold data about the variables 
    of a dynamical system, such as inputs, outputs, and states.

    name: Name of the variable (e.g., i_d, v_q, etc.)
    component: Unique name of the component associated with each state.
    type: Variable type (i.e, 'device' or 'grid')
    init: Initial conditions
    """
    name: any
    component: any = None
    type: any = None
    init: any = None

    def __post_init__(self):
        # Convert fields to NumPy arrays if they aren't already
        self.name = np.asarray(self.name)
        
        # Infer number of variables
        n = len(self.name)

        # Convert provided fields or initialize defaults
        self.component = np.asarray(self.component) if self.component is not None else np.full(n, '', dtype=str)
        self.type   = np.asarray(self.type)     if self.type is not None    else np.full(n, 'grid', dtype=str)
        self.init     = np.asarray(self.init)       if self.init is not None      else np.full(n, np.nan, dtype=float)

        # Enforce consistent lengths
        lengths = {len(self.name), len(self.component), len(self.type), len(self.init)}
        if len(lengths) != 1:
            raise ValueError("All fields must have the same length.")
            
    @property
    def n_grid(self):
        return sum(self.type == 'grid')
    
    @property
    def n_device(self):
        return sum(self.type == 'device')
        

    def __add__(self, other):
        # Concatenate to variables arrays column-wise
        return DynamicalVariables(
            name=np.concatenate([self.name, other.name]),
            component=np.concatenate([self.component, other.component]),
            type=np.concatenate([self.type, other.type]),
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
            
        return DynamicalVariables(
            name=self.name[idx],
            component=self.component[idx],
            type=self.type[idx],
            init=self.init[idx])
    
    def to_list(self):
        # Return unique a tuple uniquely identifying each variable 
        return list(zip(self.component.tolist(), self.name.tolist()))


@dataclass(slots=True)
class StateSpaceModel:
    """
    State-space representation of a dynamical system
    """
    A: np.ndarray 
    B: np.ndarray 
    C: np.ndarray 
    D: np.ndarray 
    u: DynamicalVariables = None
    y: DynamicalVariables = None
    x: DynamicalVariables = None
    
    def __post_init__(self):
        # Check that sizes match for A,B,C,D and inputs/outputs
        A_x, A_z = self.A.shape
        B_x, B_u = self.B.shape
        C_y, C_x = self.C.shape
        D_y, D_u = self.D.shape
        
        assert A_x == A_z, "A is not square."

        assert A_x == B_x, "Incorrect dimensions for A and B."
        assert A_x == C_x, "Incorrect dimensions for A and C."
        assert D_y == C_y, "Incorrect dimensions for C and D."
        assert D_u == B_u, "Incorrect dimensions for B and D."

        if self.u is None:
            self.u = DynamicalVariables(np.array([f'u{i}' for i in range(B_u)]))
        if self.y is None:
            self.y = DynamicalVariables(np.array([f'y{i}' for i in range(C_y)]))
        if self.x is None:
            self.x = DynamicalVariables(np.array([f'x{i}' for i in range(A_x)]))
            
        assert len(self.u) == B_u
        assert len(self.y) == C_y
        assert len(self.x) == A_x
        
    @property
    def data(self):
        return self.A, self.B, self.C, self.D

    @property
    def shape(self):
        return len(self.u), len(self.y), len(self.x)
    
    @classmethod
    def from_stacked(cls, components):
        """
        Create a state space-model by stacking a collection of state-space models.
        """
        fields = ['A', 'B', 'C', 'D', 'u', 'y', 'x']
        
        stack = {f: [getattr(c, f) for c in components] for f in fields}
        #stack = dict(zip(fields(StateSpaceModel), zip(*components)))
        A = block_diag(*stack['A'])
        B = block_diag(*stack['B'])
        C = block_diag(*stack['C'])
        D = block_diag(*stack['D'])
        u = sum(stack['u'], DynamicalVariables(name=[]))
        y = sum(stack['y'], DynamicalVariables(name=[]))
        x = sum(stack['x'], DynamicalVariables(name=[]))

        return cls(A=A, B=B, C=C, D=D, u=u, y=y, x=x)


    @classmethod
    def from_interconnected(cls, components, connections):
        """
        Create a state space-model by interconnecting a collection of state-space models.
        """
        F, G, H, L = connections
        sys = cls.from_stacked(components)
        I_y = np.eye(F.shape[1])
        I_u = np.eye(F.shape[0])

        A = sys.A + sys.B @ F @ np.linalg.inv(I_y - sys.D @ F) @ sys.C
        B = sys.B @ np.linalg.inv(I_u - F @ sys.D) @ G
        C = H @ np.linalg.inv(I_y - sys.D @ F ) @ sys.C
        D = H @ np.linalg.inv(I_y - sys.D @ F ) @ sys.D @ G + L
        sys.A, sys.B, sys.C, sys.D = A, B, C, D
        
        # TODO: Add support for multiplication and addition?
        sys.u = sys.u[sys.u.type == "device"]

        return sys
    
    @classmethod
    def from_csv(cls, filepath):
        pass

    def coordinate_transform(self, invT, T):
        pass

    def to_csv(self, filepath):
        # Row and column names
        u = self.u.to_list()
        y = self.y.to_list()
        x = self.x.to_list()
        # Save each matrix
        os.makedirs(filepath, exist_ok=True)
        matrix_to_csv(filepath=os.path.join(filepath, "A.csv"), matrix=self.A, index=x, columns=x)
        matrix_to_csv(filepath=os.path.join(filepath, "B.csv"), matrix=self.B, index=x, columns=u)
        matrix_to_csv(filepath=os.path.join(filepath, "C.csv"), matrix=self.C, index=y, columns=x)
        matrix_to_csv(filepath=os.path.join(filepath, "D.csv"), matrix=self.D, index=y, columns=u)

    def __repr__(self):
        return "StateSpaceModel with %d inputs, %d outputs, and %d states." % self.shape
    

def modal_analisis(A : np.ndarray, 
                   show : bool = False, 
                   print_settings : dict = {'index' : True, 
                                            'tablefmt': "grid",
                                            'numalign': "right", 
                                            'floatfmt': '.3f'}):
    '''
    Computes eigenvalues, natural frequency, damping ratio, time constant. It also has the option to display a
    pretty table when the function is executed.
    
    Args:
    ----
    A (numpy array): Matrix A of state-space model:
    
    show (Boolean): True (print table), False (do not print). By default is False.
    
    print_settings (dict): setting applied to tabulate package to print the pandas dataframe.

    Returns:
    -------

    df (Dataframe) : It contains eigenvalues, real, imag parts, natural frequency, damping ratio, and time constant.
    '''

    eigenvalues = eigvals(A)

    df = pd.DataFrame(data=eigenvalues, columns= ['eigenvalue'])
    df['real'] = df.apply(lambda row: row['eigenvalue'].real, axis=1)
    df['imag'] = df.apply(lambda row: row['eigenvalue'].imag, axis=1)
    df['natural_frequency'] = df.apply(lambda row: abs(row['eigenvalue']/(2*np.pi)), axis=1)
    df['damping_ratio'] =  df.apply(lambda row: -row['eigenvalue'].real/(abs(row['eigenvalue'])), axis=1)
    df['time_constant'] = df.apply(lambda row: -1/row['eigenvalue'].real, axis=1)
    df = df.sort_values(by='real', ascending=False, ignore_index=True)


    if show:
        df_to_print = df.copy()
        df_to_print = df_to_print[['real', 'imag', 'damping_ratio', 'natural_frequency', 'time_constant']]
        df_to_print.rename(columns={'real': 'Eigenvalue \n real part',
                                    'imag': 'Eigenvalue \n imaginary part',
                                    'damping_ratio': 'Damping \n ratio [p.u.]', 
                                    'natural_frequency': 'Natural \n frequency [Hz]',
                                    'time_constant': 'Time \n constant [s]'}, inplace=True)
        print(df_to_print.to_markdown(**print_settings))

    return df