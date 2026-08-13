import numpy as np
from scipy.linalg import solve, eig, cholesky, svd

from sting.utils.dynamical_systems import StateSpaceModel
from sting.modules.small_signal_modeling.core import SmallSignalModel
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
    B_r = B[0,0] - A[0,1]@invA_11@B[1,0]
    C_r = C[0,0] - C[0,1]@invA_11@A[1,0]
    D_r = ss.D - C[0,1]@invA_11@B[1,0]

    ss_r = StateSpaceModel(A=A_r, B=B_r, C=C_r, D=D_r)

    return ss_r
    

def get_jordan_real_transform(A:np.ndarray):
    """
    Perform a similarity transform to convert a linear state-space
    into it's modal basis. 
    
    That is, in the returned system, each state (or pair of states)
    corresponds to a model of A. A *real* Jordan form decomposition
    is used to ensure that the returned model is real valued.
    """
    d, V = eig(A)
    n = len(d)
    # Sort eigenstates from slowest to fastest 
    idx = np.argsort(np.abs(d))
    # Reorder eigenvectors and eigenvalues as needed
    T = V[:, idx]
    d = d[idx]

    # Construct transform T such that J = invT * A * T yields the
    # *real* Jordan form of A
    i = 0
    while i < n:
        # Split complex eigenvalues into real components
        if d[i].imag != 0:
            T[:, i] = T[:, i].real
            T[:, i+1] = T[:, i+1].imag
            i = i + 2
        else:
            i = i + 1

    T = T.real
    invT = solve(T, np.eye(n))

    return T, invT


def get_balancing_transform(P, Q, r:int=None):
    """
    Return the balancing transformation (or projection matrices)
    such that the controllability and observability gramians of
    the state-space model are equal and diagonal.
    """
    R = cholesky(P, lower=True)
    L = cholesky(Q, lower=True)
    
    U, sigma, Vh = svd(L.T @ R)
    V = Vh.T

    if r is None:
        # Compute the full transform
        S = np.diag(sigma**(-0.5))

        # Full similarity transformation matrices (T is square)
        T = R @ V @ S
        invT = S @ U.T @ L.T

    else:
        # Compute the projection matrices directly
        U_r = U[:, :r]
        S_r = np.diag(sigma[:r]**(-0.5))
        V_r = V[:, :r]

        # Reduced similarity transformation matrices (T_r is not square)
        T = R @ V_r @ S_r
        invT = S_r @ U_r.T @ L.T

    return T, invT