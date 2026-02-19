import numpy as np
import os
import pandas as pd
from more_itertools import transpose
from scipy.linalg import eigvals, block_diag, solve
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from sting.utils.matrix_tools import matrix_to_csv, csv_to_matrix
from typing import Self, Callable
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
        self._value = np.full(len(self._name), np.nan) if value is None else np.atleast_1d(value)
        self._time = np.atleast_1d(time) if time is not None else np.atleast_1d(np.nan)
    
    def __post_init__(self):

        for attr in self.__slots__:
            if attr in ['_name', '_time']:
                continue
            self.check_shapes(getattr(self, attr))

    # Utility methods
    # --------------------------
    
    def check_shapes(self, new_value):
        if new_value.shape[0] != self._name.shape[0]:
            raise ValueError(f"Length of attribute does not match length of 'name' ({self._name.shape[0]}).")

    def to_list(self):
        # Return unique a tuple uniquely identifying each variable
        return list(zip(self.component.tolist(), self.name.tolist()))
    
    def to_dataframe(self, csv_filepath = None):
        fields = list(self.__slots__)
        fields.remove('_time')
        fields.remove('_value')
        d = {k.lstrip('_'): getattr(self, k) for k in fields}
        df = pl.DataFrame(d)

        if csv_filepath is not None:
            df.write_csv(csv_filepath)
    
        return df

    def to_timeseries(self):
        d = {k : self._value[i] for i, k in enumerate(self._name)}
        df = pl.DataFrame(d)
        new_col = pl.Series("time", self._time)
        df = df.insert_column(0, new_col)
        return df

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
        new_value = np.atleast_1d(new_value).astype(float)
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
    
    # Special methods
    # --------------------------
    def __len__(self):
        return len(self.name)
    
    def __add__(self, other):
        # Concatenate to variables arrays column-wise
        if not np.array_equal(self.time, other.time, equal_nan=True):
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
            self.u = DynamicalVariables(np.array([f"u{i}" for i in range(B_u)]))
        if self.y is None:
            self.y = DynamicalVariables(np.array([f"y{i}" for i in range(C_y)]))
        if self.x is None:
            self.x = DynamicalVariables(np.array([f"x{i}" for i in range(A_x)]))

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
    def from_stacked(cls, components: list[Self]):
        """
        Create a state space-model by stacking a collection of state-space models.
        """
        fields = ["A", "B", "C", "D", "u", "y", "x"]
        selection = [[getattr(c, f) for f in fields] for c in components]
        
        stack = dict(zip(fields, transpose(selection)))
        A = block_diag(*stack["A"])
        B = block_diag(*stack["B"])
        C = block_diag(*stack["C"])
        D = block_diag(*stack["D"])
        u = sum(stack["u"], DynamicalVariables(name=[]))
        y = sum(stack["y"], DynamicalVariables(name=[]))
        x = sum(stack["x"], DynamicalVariables(name=[]))

        return cls(A=A, B=B, C=C, D=D, u=u, y=y, x=x)
   
    @classmethod
    def from_interconnected(cls, 
                             components: list[Self], 
                             connections: list[np.ndarray], 
                             u: DynamicalVariables | Callable[[DynamicalVariables], DynamicalVariables],
                             y: DynamicalVariables | Callable[[DynamicalVariables], DynamicalVariables],
                             component_label: str = None):
        
        F, G, H, L = connections
        sys = cls.from_stacked(components)
        I_y = np.eye(F.shape[1])
        I_u = np.eye(F.shape[0])

        A = sys.A + sys.B @ F @ np.linalg.inv(I_y - sys.D @ F) @ sys.C
        B = sys.B @ np.linalg.inv(I_u - F @ sys.D) @ G
        C = H @ np.linalg.inv(I_y - sys.D @ F) @ sys.C
        D = H @ np.linalg.inv(I_y - sys.D @ F) @ sys.D @ G + L
        
        u = u if not callable(u) else u(sys.u)
        y = y if not callable(y) else y(sys.y)

        new_sys = cls(A=A, B=B, C=C, D=D, u=u, y=y, x=sys.x)
        
        # TODO: Add support for multiplication and addition?
        if component_label is not None:
            new_sys.x.component = component_label
            new_sys.u.component = component_label
            new_sys.y.component = component_label

        return new_sys   

    @classmethod
    def from_csv(cls, filepath):
        A, x, _ = csv_to_matrix(os.path.join(filepath, "A.csv"))
        B, _, _ = csv_to_matrix(os.path.join(filepath, "B.csv"))
        C, _, _ = csv_to_matrix(os.path.join(filepath, "C.csv"))
        D, y, u = csv_to_matrix(os.path.join(filepath, "D.csv"))

        x = tuple(map(list, zip(*x)))
        x = DynamicalVariables(component=x[0], name=x[1])

        y = tuple(map(list, zip(*y)))
        y = DynamicalVariables(component=y[0], name=y[1])

        u = tuple(map(list, zip(*u)))
        u = DynamicalVariables(component=u[0], name=u[1])

        return cls(A=A, B=B, C=C, D=D, x=x, y=y, u=u)

    def coordinate_transform(self, invT, T):
        pass

    def to_csv(self, filepath):
        
        # Create output directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)

        # Export variables
        self.x.to_dataframe(os.path.join(filepath, "x.csv"))
        self.u.to_dataframe(os.path.join(filepath, "u.csv"))
        self.y.to_dataframe(os.path.join(filepath, "y.csv"))

        # Row and column names
        u = self.u.to_list()
        y = self.y.to_list()
        x = self.x.to_list()
        
        # Export each matrix
        matrix_to_csv(
            filepath=os.path.join(filepath, "A.csv"), matrix=self.A, index=x, columns=x
        )
        matrix_to_csv(
            filepath=os.path.join(filepath, "B.csv"), matrix=self.B, index=x, columns=u
        )
        matrix_to_csv(
            filepath=os.path.join(filepath, "C.csv"), matrix=self.C, index=y, columns=x
        )
        matrix_to_csv(
            filepath=os.path.join(filepath, "D.csv"), matrix=self.D, index=y, columns=u
        )

    def __repr__(self):
        return "StateSpaceModel with %d inputs, %d outputs, and %d states." % self.shape
    
    def sim(self, tps, u_func, x0 = None, ode_method= 'Radau', ode_max_step = 0.01):

        if x0 is None:
            x0 = np.zeros_like(self.x.init)

        def state_space_ode(t, x, u_func):
            """
            Defines the right-hand side of the state-space differential equation.

            Args:
            t (float): Current time.
            x (np.ndarray): Current state vector.
            A (np.ndarray): State matrix.
            B (np.ndarray): Input matrix.
            u_func (callable): Function that returns the input vector u at time t.

            Returns:
            np.ndarray: Time derivative of the state vector (dx/dt).
            """

            u = u_func(t)
            return self.A @ x + self.B @ u
        
        t_in = tps[0]
        t_fin = tps[-1]
        sol = solve_ivp(
                        fun=state_space_ode,
                        t_span=[t_in, t_fin],
                        y0=x0,
                        args=(u_func,),
                        method = ode_method,
                        max_step = ode_max_step,
                        dense_output=True # To get a continuous solution for plotting
                        )
        interp_sol = sol.sol(tps)

        return interp_sol
    
    def modal_analysis(
        self,
        show: bool = False,
        print_settings: dict = {
            "index": True,
            "tablefmt": "grid",
            "numalign": "right",
            "floatfmt": ".3f",
        },
    ):
        """
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
        """

        eigenvalues = eigvals(self.A)

        df = pd.DataFrame(data=eigenvalues, columns=["eigenvalue"])
        df["real"] = df.apply(lambda row: row["eigenvalue"].real, axis=1)
        df["imag"] = df.apply(lambda row: row["eigenvalue"].imag, axis=1)
        df["natural_frequency"] = df.apply(
            lambda row: abs(row["eigenvalue"] / (2 * np.pi)), axis=1
        )
        df["damping_ratio"] = df.apply(
            lambda row: -row["eigenvalue"].real / (abs(row["eigenvalue"])), axis=1
        )
        df["time_constant"] = df.apply(lambda row: -1 / row["eigenvalue"].real, axis=1)
        df = df.sort_values(by="real", ascending=False, ignore_index=True)

        if show:
            df_to_print = df.copy()
            df_to_print = df_to_print[
                ["real", "imag", "damping_ratio", "natural_frequency", "time_constant"]
            ]
            df_to_print.rename(
                columns={
                    "real": "Eigenvalue \n real part",
                    "imag": "Eigenvalue \n imaginary part",
                    "damping_ratio": "Damping \n ratio [p.u.]",
                    "natural_frequency": "Natural \n frequency [Hz]",
                    "time_constant": "Time \n constant [s]",
                },
                inplace=True,
            )
            print(df_to_print.to_markdown(**print_settings))

        return df
    
    def coordinate_transform(self, T:np.ndarray, invT:np.ndarray): #-> StateSpaceModel:
        """Perform a coordinate transformation z = Tx (analogous to MATLAB ss2ss)"""
        A_t = invT @ self.A @ T
        B_t = invT @ self.B
        C_t = self.C @ T
        return StateSpaceModel(A=A_t, B=B_t, C=C_t, D=self.D)
          