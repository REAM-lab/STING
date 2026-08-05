import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag

from sting.system.core import System
from sting.utils.component_connections import build_ccm_permutation, get_ccm_matrices
from sting.utils.dynamical_systems import DynamicalVariables


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
        I_y = np.eye(L_11.shape[1])
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
        C = L_21@sys.C + L_12@sys.D@inv@L_11@sys.C
        D = L_21@sys.D@inv@L_12 + L_22

        u = u if not callable(u) else u(sys.u)
        y = y if not callable(y) else y(sys.y)

        new_sys = cls(A=A, B=B, C=C, D=D, H=H, N=N, u=u, y=y, x=sys.x)

        if component_label is not None:
            new_sys.x.component = component_label
            new_sys.u.component = component_label
            new_sys.y.component = component_label

        return new_sys

    def from_system(cls, system:System, power_flow_solution):
        # Load all components that are compatible with the component connection method
        components = system.query("ccm_generators", "ccm_shunts", "ccm_branches").to_list()

        # Construct component quadratic bilinear models
        for c in components:
            c._calculate_emt_initial_conditions()
        for c in components:
            c._build_quadratic_bilinear_model()
        models = [c.qbm for c in components]

        # Construct interconnection matrices
        L11, L12, L21, L22 = get_ccm_matrices(system, attribute="ssm", dimI=2)
        # Permute the F and G 
        T = build_ccm_permutation(system)
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
            x0: list[float] = None, 
            settings={'dense_output': True, 'method': 'Radau', 'max_step': 0.001}):
    
            if x0 is None:
                x0 = self.x.init
    
            if inputs is None:
                inputs = {}
            inputs = self.vectorize_inputs(inputs)
                   
            sol = solve_ivp(
                fun=self.get_derivatives_step,
                t_span=[0, t_max],
                y0=x0,
                dense_output=settings['dense_output'],  
                args=(inputs, ),
                method=settings['method'], 
                max_step=settings['max_step'])
                            
            # Define timepoints that will be used to evaluate the solution of the ODEs
            if settings['dense_output']:
                tps = np.linspace(0, t_max, 500)
                sol = sol.sol(tps)

            sol.x = sol.y
            sol.u = np.array([inputs(t) for t in sol.t])
            sol.y = self.C@sol.x + sol.D@sol.u

            return tps, sol


    def shift_to_equilibrium(self):
        """Center the dynamics of model about its initial conditions"""
        n, m = self.B.shape

        x0 = self.x.init.reshape(-1, 1)
        u0 = self.u.init.reshape(-1, 1)

        K1 = kronecker_commute(n,n)
        K2 = kronecker_commute(n,m)

        self.A = (
            self.A 
            + self.H @ (K1 + np.eye(n**2)) @ np.kron(x0, np.eye(n)) 
            + self.N @ np.kron(u0, np.eye(n))
        )
        self.B = (
            self.B 
            + self.N @ K2 @ np.kron(x0, np.eye(m)) 
        )

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