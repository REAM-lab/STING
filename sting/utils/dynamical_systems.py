# ---------------------------------------
# Import standard and third-party packages
# ----------------------------------------
import logging
import os
from dataclasses import dataclass
from typing import Callable, Self, Literal

import matplotlib
import numpy as np
import plotly.graph_objects as go
import polars as pl
import pylab as plt
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, eigvals, solve_continuous_lyapunov, cholesky
from scipy.linalg.lapack import dpstrf
matplotlib.use("Agg")

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag

from sting.utils.component_connections import build_ccm_permutation, get_ccm_matrices

# --------------
# Import sting code
# --------------
from sting.utils.matrix_tools import csv_to_matrix, matrix_to_csv

# Set up logger
logger = logging.getLogger(__name__)

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

    def to_timeseries(self, csv_filepath = None):
        d = {k : self._value[i] for i, k in enumerate(self._name)}
        df = pl.DataFrame(d)
        new_col = pl.Series("time", self._time)
        df = df.insert_column(0, new_col)
        if csv_filepath is not None:
            df.write_csv(csv_filepath)
        return df
    
    def to_plotly(self, figure_filepath = None):
        """Plot the dynamical variables using plotly. It creates a subplot for each variable, with shared x-axis. 
        The figure is saved as an html file if figure_filepath is provided."""
        
        # Create two columns with shared x-axis
        ncols = 2
        nrows = len(self._name) // ncols + int(len(self._name) % ncols > 0)
        
        fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True)
        for i in range(len(self._name)):
            fig.add_trace(go.Scatter(x=self._time, y=self._value[i], name=self._name[i]), row=i//ncols+1, col=i%ncols+1)
            fig.update_yaxes(title_text=self._name[i], row=i//ncols+1, col=i%ncols+1)
            fig.update_xaxes(title_text='Time [s]',row=i//ncols+1, col=i%ncols+1)

        fig.update_layout(title_text = self._component[0], title_x=0.5, showlegend = False, height=300*nrows)
        
        if figure_filepath is not None:
            fig.write_html(figure_filepath)

        return fig
        

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
        
        stack = dict(zip(fields, zip(*selection)))
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

    def to_quadratic_bilinear(self):
        # Import here to avoid potential circular imports
        # from .quadratic_bilinear_model import QuadraticBilinearModel

        n, m = self.B.shape
        # State-space model is a quadratic bilinear model with H = 0 and N = 0
        H = np.zeros((n, n*n))
        N = np.zeros((n, n*m))

        return QuadraticBilinearModel(A=self.A, B=self.B, C=self.C, D=self.D, H=H, N=N, x=self.x, y=self.y, u=self.u)

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
    
    def simulate(
            self, 
            t_max: float, 
            inputs: dict[str, dict[str, Callable[[float], float]]] = None, 
            x0: list[float] = None, 
            settings={'dense_output': True, 'method': 'Radau', 'max_step': 0.001},
            output_directory: str = os.getcwd(), 
            plot: bool = True):

        if x0 is None:
            x0 = self.x.init

        if inputs is None:
            inputs = {}

        def state_space_ode(t: float, x: np.ndarray,  inputs: dict[str, dict[str, Callable[[float], float]]]):
            """
            Defines the right-hand side of the state-space differential equation.

            Args:
            t (float): Current time.
            x (np.ndarray): Current state vector.
            A (np.ndarray): State matrix.
            B (np.ndarray): Input matrix.
            inputs (dict): Dictionary of input functions indexed by component and name.

            Returns:
            np.ndarray: Time derivative of the state vector (dx/dt).
            """

            u = [inputs[component][name](t) if inputs.get(component, {}).get(name) else 0.0 for (component, name) in zip(self.u.component, self.u.name)]
            return self.A @ x + self.B @ u
               
        solution = solve_ivp(
                        fun=state_space_ode,
                        t_span=[0, t_max],
                        y0=x0,
                        dense_output=settings['dense_output'],  
                        args=(inputs, ),
                        method=settings['method'], 
                        max_step=settings['max_step'])
                        
        # Define timepoints that will be used to evaluate the solution of the ODEs
        if settings['dense_output']:
            tps = np.linspace(0, t_max, 500)
            solution = solution.sol(tps)

        if plot:
            self.plot_simulation(output_directory=output_directory, tps=tps, solution=solution)
        
        return tps, solution
    
    def plot_simulation(self, output_directory: str, tps: np.ndarray, solution):

        number_of_states = self.shape[2]
        nrows = int(np.ceil(number_of_states / 2))
        ncols = 2 if number_of_states > 1 else 1

        fig = make_subplots(rows=nrows, cols=ncols)

        for i in range(number_of_states):
            row = i // ncols + 1
            col = i % ncols + 1
            fig.add_trace(go.Scatter(x=tps, y=solution[i]), row=row, col=col)
            fig.update_xaxes(title_text='Time [s]', row=row, col=col)
            fig.update_yaxes(title_text=self.x.name[i], row=row, col=col)
        
        fig.write_html(os.path.join(output_directory, "simulation.html"))
    
    def modal_analysis(self):
        """
        Computes eigenvalues, natural frequency, damping ratio, time constant. It also has the option to display a
        pretty table when the function is executed.

        Args:
        ----
        A (numpy array): Matrix A of state-space model:

        show (Boolean): True (print table), False (do not print). By default is False.

        Returns:
        -------

        df (Dataframe) : It contains eigenvalues, real, imag parts, natural frequency, damping ratio, and time constant.
        """

        eigenvalues = eigvals(self.A)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        df = pl.DataFrame(data=(real_parts, imag_parts), schema=["real", "imag"])
        df = df.with_columns( ((pl.col("real")**2 + pl.col("imag")**2).sqrt()).alias("magnitude") )        
        df = df.with_columns( (pl.col("magnitude")/(2*np.pi)).alias("natural_frequency_hz") )
        df = df.with_columns( (-pl.col("real")/pl.col("magnitude")).alias("damping_ratio_pu") )
        df = df.with_columns( (-1/pl.col("real")).alias("time_constant_seconds") )
        df = df.drop("magnitude")
        df = df.sort("real", descending=True)     

        df_to_print = df.with_columns(pl.col("real").round(3), 
                                      pl.col("imag").round(3), 
                                      pl.col("natural_frequency_hz").round(3), 
                                      pl.col("damping_ratio_pu").round(3), 
                                      pl.col("time_constant_seconds").round(4)) 
        logger.info("Modal analysis results:")
        logger.info(df_to_print)

        return df
    
    def plot_eigenvalues(self, ax=None, **kwargs):
        eigenvalues = eigvals(self.A)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        if ax is None:
            _, ax = plt.subplots(1,1, figsize=(8, 6))

        ax.scatter(x=real_parts, y=imag_parts, **kwargs)

        # Check if the x-axis label is empty and set it
        if not ax.get_xlabel():
            ax.set_xlabel(r'$\Re(\lambda)$') #

        # Check if the y-axis label is empty and set it
        if not ax.get_ylabel():
            ax.set_ylabel(r'$\Im(\lambda)$') #

        return ax
    
    def coordinate_transform(self, T:np.ndarray, invT:np.ndarray): #-> StateSpaceModel:
        """Perform a coordinate transformation z = Tx (analogous to MATLAB ss2ss)"""
        A_t = invT @ self.A @ T
        B_t = invT @ self.B
        C_t = self.C @ T
        # Compute the new states initial conditions
        x = DynamicalVariables(name=[f"x{i}" for i in range(T.shape[1])], init=invT@self.x.init)

        return StateSpaceModel(A=A_t, B=B_t, C=C_t, D=self.D, x=x, u=self.u, y=self.y)


    def gramian(self, kind: Literal["controllability", "observability"]):
        """
        Returns the Gramian of the state-space model.

        Parameters
        ----------
        kind: Whether to compute the "controllability" or "observability" Gramian

        cholesky: If True returns the Cholesky factorization of the Gramians

        lower: Only applicable if `cholesky=True`, if True will return the 
            lower Cholesky factorization of the Gramian.
        """
        match kind: 
            case "controllability":
                W = solve_continuous_lyapunov(self.A, -self.B@self.B.T)
            case "observability":
                W = solve_continuous_lyapunov(self.A.T, -self.C.T@self.C)

        return W


# -------------
# Functions
# -------------
def smooth_step(t: float, step_time: float, initial_value: float, final_value: float, transient_width: float) -> float:
    """
    Returns the value of a smooth step function at time t. 
    The function transitions from initial_value to final_value around step_time with a transition width defined by transient_width.
    
    Inputs:
    - t: time at which to evaluate the function
    - step_time: time at which the step occurs
    - initial_value: value of the function before the step
    - final_value: value of the function after the step
    - transient_width: width of the transition region (the larger, the smoother the transition). It should be a bit larger than integration step, e.g. 1e-3 for a 1e-4 integration step.

    Outputs
    - The value of the smooth step function at time t.
    """
    return initial_value + (final_value - initial_value) * 0.5 * (1 + np.tanh((t - step_time)/transient_width))



def make_smooth_step(step_time: float, initial_value: float, final_value: float, transient_width: float):
    return lambda t: smooth_step(t, step_time, initial_value, final_value, transient_width)


@dataclass
class QuadraticBilinearModel:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    H: np.ndarray
    N: np.ndarray
    
    x: DynamicalVariables = None
    u: DynamicalVariables = None
    y: DynamicalVariables = None

    def __post_init__(self):
        pass


    @classmethod
    def from_stacked(cls, components: list[Callable]):
        """
        Create a state space-model by stacking a collection of state-space models.
        """
        fields = ["A", "B", "C", "D", "H", "N", "u", "y", "x"]
        selection = [[getattr(c, f) for f in fields] for c in components]
        
        stack = dict(zip(fields, zip(*selection)))
        A = block_diag(*stack["A"])
        B = block_diag(*stack["B"])
        C = block_diag(*stack["C"])
        D = block_diag(*stack["D"])

        H, N = [], []
        for B_i, H_i, N_i in zip(stack["B"], stack["H"], stack["N"]):
            n, m = B_i.shape
            H.append(np.stack(np.hsplit(H_i, n), axis=-1))
            N.append(np.stack(np.hsplit(N_i, m), axis=-1))

        H = cube_diag(*H)
        N = cube_diag(*N)

        u = sum(stack["u"], DynamicalVariables(name=[]))
        y = sum(stack["y"], DynamicalVariables(name=[]))
        x = sum(stack["x"], DynamicalVariables(name=[]))

        return cls(A=A, B=B, C=C, D=D, H=H, N=N, u=u, y=y, x=x)

    @classmethod
    def from_interconnected(cls, components, connections, u, y, component_label:str=None):
        """
        a = u_stack, b = y_stack

        a = L11 * b + L12*u + M1 (x otimes x) + M2 (u otimes x)
        y = L21 * b + L22*u
        """
        (L_11, L_12, L_21, L_22, M_1, M_2) = connections
        
        sys = cls.from_stacked(components)
        I_y = np.eye(L_11.shape[0])
        n, _ = sys.A.shape

        inv = np.linalg.inv(I_y - L_11@sys.D)

        if (M_1 is None) and (M_2 is None):
            # If M1 and M2 are None, then they are assumed to be
            # zeros and we can remove X and Y from our system level model
            X = 0
            Y = 0
        else:
            # All of these matrices should be zero
            assert np.all(sys.N@np.kron(inv@M_1, np.eye(n)) == 0)
            assert np.all(sys.N@np.kron(inv@M_2, np.eye(n)) == 0)
            assert np.all(L_21@sys.D@inv@M_1 == 0)
            assert np.all(L_21@sys.D@inv@M_2 == 0)

            X = sys.B@inv@M_1
            Y = sys.B@inv@M_2 
            
        A = sys.A + sys.B@inv@L_11@sys.C 
        H = sys.H + X + sys.N@np.kron(inv@L_11@sys.C, np.eye(n))
        B = sys.B@inv@L_12
        N = Y + sys.N@np.kron(inv@L_12, np.eye(n))
        C = L_21@sys.C + L_21@sys.D@inv@L_11@sys.C
        D = L_21@sys.D@inv@L_12 + L_22

        u = u if not callable(u) else u(sys.u)
        y = y if not callable(y) else y(sys.y)

        new_sys = cls(A=A, B=B, C=C, D=D, H=H, N=N, u=u, y=y, x=sys.x)

        if component_label is not None:
            new_sys.x.component = component_label
            new_sys.u.component = component_label
            new_sys.y.component = component_label

        return new_sys

    @classmethod
    def from_system(cls, system, power_flow_solution, timepoint=None):
        # Load all components that are compatible with the component connection method
        components = system.query(["ccm_generators", "ccm_shunts", "ccm_branches"]).to_list()

        # Load the ACOPF solution into each component
        if timepoint is None:
            t = system.timepoints[0]
        for c in components:
            c.load_ac_power_flow_solution(t.name, power_flow_solution)

        # Construct component quadratic bilinear models
        for c in components:
            c._calculate_emt_initial_conditions()
        for c in components:
            c._build_quadratic_bilinear_model()
        models = [c.qbm for c in components]

        # Construct interconnection matrices
        L11, L12, L21, L22 = get_ccm_matrices(system, attribute="qbm", dimI=2)
        # Permute the F and G 
        T = build_ccm_permutation(system, attribute="qbm")
        T = block_diag(T, np.eye(L11.shape[0] - T.shape[0]))
        L11 = T @ L11
        L12 = T @ L12

        # Construct system level model
        connections = (L11, L12, L21, L22, None, None)
        u = lambda u: u[u.type == "device"] # System inputs are device inputs
        y = lambda y: y # System outputs are all component outputs
        # Interconnect models
        qbm = cls.from_interconnected(models, connections, u, y)

        return qbm


    def get_derivatives_step(self, t: float, x: np.ndarray,  inputs: Callable):
        u = inputs(t)
        dx = self.A@x + self.H@np.kron(x,x) + self.N@np.kron(u, x) + self.B@u
        return dx


    def simulate(
        self, 
        t_max: float, 
        inputs: dict[str, dict[str, Callable[[float], float]]] = None, 
        settings={'dense_output': True, 'method': 'Radau', 'max_step': 0.001},
        shift=False):

        if shift:
            x0 = np.zeros_like(self.x.init)
            u0 = np.zeros_like(self.u.init)
            qbm = self.shift_to_equilibrium()
            #x_offset = self.x.init
            #u_offset = self.u.init

        else:
            x0 = self.x.init
            u0 = self.u.init
            qbm = self

        inputs_to_sim = lambda t: self.vectorize_inputs(inputs)(t) + u0
                
        sol = solve_ivp(
            fun=qbm.get_derivatives_step,
            t_span=[0, t_max],
            y0=x0,
            dense_output=settings['dense_output'],  
            args=(inputs_to_sim, ),
            method=settings['method'], 
            max_step=settings['max_step'])
                        
        # Define timepoints that will be used to evaluate the solution of the ODEs
        if settings['dense_output']:
            tps = np.linspace(0, t_max, 500)
            sol.y = sol.sol(tps)
            sol.t = tps

        sol.x = sol.y
        sol.u = np.array([inputs_to_sim(t) for t in sol.t]).T
        sol.y = self.C@sol.x + self.D@sol.u

        return sol

    def write_simulation_csv(self, solution, output_directory):       
        # Get the components in the same order as solution vector
        _, comp_idx = np.unique(self.x.component, return_index=True)
        components = self.x.component[np.sort(comp_idx)]  

        # Write the simulation results to CSV files.
        i = 0
        for component in components:
            number_of_states = sum(self.x.component == component)
            state_names = self.x.name[self.x.component == component]
            columns_for_df = ['time'] + state_names.tolist()
            (pl.DataFrame(
                data=np.column_stack((solution.t, solution.x[i:i+number_of_states].T)),
                schema=columns_for_df
            )
            .write_csv(os.path.join(output_directory, f"{component}.csv"))
            )
            i += number_of_states

    def write_simulation_plots(self, solution, output_directory):

         # Get the components in the same order as solution vector
        _, comp_idx = np.unique(self.x.component, return_index=True)
        components = self.x.component[np.sort(comp_idx)] 
        
        # Make a html file for each component. Each file plots the states corresponding to each component.
        i = 0
        for component in components:
            number_of_states = sum(self.x.component == component)
            nrows = int(np.ceil(number_of_states / 2))
            ncols = 2 if number_of_states > 1 else 1
            fig = make_subplots(rows=nrows, cols=ncols)
            for j in range(number_of_states):
                row = j // ncols + 1
                col = j % ncols + 1
                fig.add_trace(go.Scatter(x=solution.t, y=solution.x[i]), row=row, col=col)
                fig.update_xaxes(title_text='Time [s]', row=row, col=col)
                fig.update_yaxes(title_text=self.x.name[i], row=row, col=col)
                i += 1

            fig.update_layout(title_text = component, title_x=0.5, showlegend = False, height=300*nrows)
            fig.write_html(os.path.join(output_directory, f"{component}.html"))

    def write_csv(self, filepath):
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
        #matrix_to_csv(
        #    filepath=os.path.join(filepath, "H.csv"), matrix=self.H, index=x, columns=None
        #)
        #matrix_to_csv(
        #    filepath=os.path.join(filepath, "N.csv"), matrix=self.N, index=x, columns=None
        #)

    def shift_to_equilibrium(self):
        """Center the dynamics of model about its initial conditions"""
        n, m = self.B.shape

        x0 = self.x.init.reshape(-1, 1)
        u0 = self.u.init.reshape(-1, 1)

        K1 = kronecker_commute(n,n)
        K2 = kronecker_commute(n,m)

        A = (
            self.A 
            + self.H @ (K1 + np.eye(n**2)) @ np.kron(x0, np.eye(n)) 
            + self.N @ np.kron(u0, np.eye(n))
        )
        B = (
            self.B 
            + self.N @ K2 @ np.kron(x0, np.eye(m)) 
        )

        return QuadraticBilinearModel(A=A, B=B, C=self.C, D=self.D, N=self.N, H=self.H, x=self.x, u=self.u, y=self.y)

    def get_symmetrized(self):
        """Return a new model where H is symmetrized"""
        n, _ = self.H.shape
        H2 = self.get_matricized_H(2)
        H3 = self.get_matricized_H(3) 

        H2_new = 0.5 * (H2 + H3)
        H1_new = np.hstack([H_i.T for H_i in np.hsplit(H2_new, n)])

        qbm_new = copy.deepcopy(self)
        qbm_new.H = H1_new

        return qbm_new

    def get_matricized_H(self, mode):
        """Return a mode-i matricization of H"""
        H = self.H
        n, _ = H.shape
        match mode:
            case 1:
                return H
            case 2:
                return np.hstack([H_i.T for H_i in np.hsplit(H, n)])
            case 3:
                return np.hstack([H_i.flatten(order="F").reshape(-1, 1) for H_i in np.hsplit(self.H, n)]).T


    def pg_project(self, W, V):
        A = W@self.A@V
        B = W@self.B
        C = self.C@V

        H = W@self.H@np.kron(V,V)
        N = W@self.N@np.kron(np.eye(self.B.shape[1]),V)

        x0 = W@self.x0

        return #QuadraticBilinearModel(A, B, C, self.D, H, N, M, x0=x0, y0=self.y0, u0=self.u0, post_solve=self.post_solve)

        
    def vectorize_inputs(self, inputs):
        return lambda t: [inputs[component][name](t) if inputs.get(component, {}).get(name) else 0.0 for (component, name) in zip(self.u.component, self.u.name)]

        

# -------------------------------------------
# Helper functions
# -------------------------------------------


def kronecker_commute(m, n):
    """
    Returns an mn x mn Kronecker commutation matrix K_(m,n)
    such that $K_{(m,n)} (x_1 otimes x_2) = (x_2 otimes x_1)$ where
    $x_1 in R^m, x_2 in R^n$.
    """
    # Swap the first two axes of the identity matrix and flatten back
    K = np.eye(m * n).reshape((m, n, m, n)).transpose(1, 0, 2, 3).reshape((m * n, m * n))
    return K

def cube_diag(*arrays):
    """
    Stack tensors into a 3D array corner to corner, then flatten result into a mode 1 matricization
    """

    shape = np.sum([a.shape for a in arrays], axis=0)
    out = np.zeros(shape, dtype=np.result_type(*arrays))

    offset = np.zeros(3, dtype=int)

    for a in arrays:
        slices = tuple(
            slice(offset[d], offset[d] + a.shape[d])
            for d in range(3)
        )
        out[slices] = a
        offset += a.shape

    # Mode 1 matricization
    out = np.hstack([out[:,:,i] for i in range(out.shape[2])])

    return out
