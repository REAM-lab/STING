import numpy as np
from scipy.linalg import solve

from sting.utils.dynamical_systems import StateSpaceModel
from sting.utils.matrix_tools import mat2cell

def singular_perturbation(ss:StateSpaceModel, r:int) -> StateSpaceModel:
    """
    Return a reduced-order model by substituting the quasi-steady state
    model into the full-order dynamics.
    """
    n = ss.A.shape[0] # Order of the sstem
    split = [r, n-r]   # Number of states in each partition
    
    # Partition the state-space matrices
    A = mat2cell(ss.A, split, split)
    B = mat2cell(ss.B, split, None)
    C = mat2cell(ss.C, None, split)
    invA_11 = solve(A[1,1], np.eye(n-r))

    # Substituting QSS model into slow dynamics
    A_r = A[0,0] - A[0,1]@invA_11@A[1,0]
    B_r = B[0] - A[0,1]@invA_11@B[1]
    C_r = C[0] - C[1]@invA_11@A[1,0]
    D_r = ss.D - C[1]@invA_11@B[1]

    ss_r = StateSpaceModel(A=A_r, B=B_r, C=C_r, D=D_r)

    return ss_r
