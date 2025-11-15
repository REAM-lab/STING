import numpy as np
import os
from sting.models.Variables import Variables
from scipy.linalg import block_diag 
from dataclasses import dataclass, fields
from sting.utils.data_tools import matrix_to_csv

@dataclass(slots=True)
class StateSpaceModel:
    """
    State-space representation of a dynamical system
    """
    A: np.ndarray 
    B: np.ndarray 
    C: np.ndarray 
    D: np.ndarray 
    u: Variables = None
    y: Variables = None
    x: Variables = None
    
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
            self.u = Variables(np.array([f'u{i}' for i in range(B_u)]))
        if self.y is None:
            self.y = Variables(np.array([f'y{i}' for i in range(C_y)]))
        if self.x is None:
            self.x = Variables(np.array([f'x{i}' for i in range(A_x)]))
            
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
        component_ssm = [c.ssm for c in components if hasattr(c, "ssm")]
        stack = {f.name: [getattr(c, f.name) for c in component_ssm] for f in fields(component_ssm[0])}
        #stack = dict(zip(fields(StateSpaceModel), zip(*components)))
        A = block_diag(*stack['A'])
        B = block_diag(*stack['B'])
        C = block_diag(*stack['C'])
        D = block_diag(*stack['D'])
        u = sum(stack['u'], Variables(name=[]))
        y = sum(stack['y'], Variables(name=[]))
        x = sum(stack['x'], Variables(name=[]))

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
        matrix_to_csv(filepath=os.path.join(filepath, "A.csv"), index=x, columns=x)
        matrix_to_csv(filepath=os.path.join(filepath, "B.csv"), index=x, columns=u)
        matrix_to_csv(filepath=os.path.join(filepath, "C.csv"), index=y, columns=x)
        matrix_to_csv(filepath=os.path.join(filepath, "D.csv"), index=y, columns=u)

    def __repr__(self):
        return "StateSpaceModel with %d inputs, %d outputs, and %d states." % self.shape