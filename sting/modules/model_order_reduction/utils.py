import warnings

import numpy as np
from scipy.linalg import cholesky, eig, solve, svd
from slycot import sb03od

from sting.modules.small_signal_modeling.core import SmallSignalModel
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel
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


    ss_r = StateSpaceModel(A=A_r, B=B_r, C=C_r, D=D_r, y=ss.y, u=ss.u, x=ss.x[:r])

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


def get_balancing_transform(P, Q, r:int=None, R=None, L=None):
    """
    Return the balancing transformation (or projection matrices)
    such that the controllability and observability gramians of
    the state-space model are equal and diagonal.
    """
    if R is None:
        R = cholesky(P, lower=True)
    if L is None:
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


def controllability_cholesky(A, B, lower=False):
    """
    Compute a factor R such that
        P = R.T @ R

    where P solves
        A @ P + P @ A.T + B @ B.T = 0.
    """
    n, m = B.shape
    
    # Store B.T in the first m rows
    B_pad = np.zeros((n, n))
    B_pad[:m, :n] = B.T

    X, scale, w = sb03od(
        n=n,
        m=m,
        A=A.T.copy(),
        Q=np.zeros_like(A),
        B=B_pad,
        dico='C',
        fact='N',
        trans='N',
    )
    if scale != 1:
        warnings.warn(f"[!] Warning scale = {scale}. Gramian may not be well conditioned.")

    if lower:
        X = X.T

    return X, scale


def observability_cholesky(A, C, lower=False):
    """
    Compute a factor L such that
        Q = L.T @ L

    where Q solves
        A.T @ Q + Q @ A + C.T @ C = 0.
    """

    p, n = C.shape

    C_pad = np.zeros((n, n))
    C_pad[:n, :p] = C.T

    X, scale, w = sb03od(
        n,
        p,
        A.copy(),
        np.zeros_like(A),
        C_pad.T.copy(),
        dico='C',
        fact='N',
        trans='N',
    )

    if scale != 1:
        warnings.warn(f"[!] Warning scale = {scale}. Gramian may not be well conditioned.")
    if lower:
        X = X.T

    return X, scale