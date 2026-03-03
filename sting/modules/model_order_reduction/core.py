# ----------------------
# Import python packages
# ----------------------
from typing import Literal, NamedTuple
from dataclasses import dataclass, field

from scipy.linalg import eig, solve, cholesky, svd
import numpy as np
# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import StateSpaceModel
from sting.modules.model_order_reduction.utils import singular_perturbation

class LinearSystem(NamedTuple):
    state_space: StateSpaceModel
    controllability_gram: np.ndarray = None
    observability_gram: np.ndarray = None

@dataclass(slots=True)
class SingularPerturbation:
    r: int 
    reduction_basis: Literal["eigen", "none"] = "eigen"

    def reduce(self, sys:LinearSystem) -> StateSpaceModel:
        """Return a reduced-order model."""
        # Perform a coordinate transform to induce timescale separation
        match self.reduction_basis:
            case "eigen":
                ss = self._to_eigenbasis(sys.state_space)
            case "none":
                ss = sys.state_space

        # Compute the ROM
        ss_r = singular_perturbation(sys=ss, r=self.r)
        
        return ss_r
        
    def _to_eigenbasis(ss:StateSpaceModel) -> StateSpaceModel:
        """
        Perform a similarity transform to convert a linear state-space
        into it's modal basis. 
        
        That is, in the returned system, each state (or pair of states)
        corresponds to a model of A. A *real* Jordan form decomposition
        is used to ensure that the returned model is real valued.
        """
        d, V = eig(ss.A)
        n = len(d)
        # Sort eigenstates from slowest to fastest 
        idx = np.argsort(np.abs(d))
        # Reorder eigenvectors and eigenvalues as needed
        T = V[:, idx]
        d = d[idx]

        # Construct transform T such that J = invT * A * T yields the
        # *real* Jordan form of A
        i = 0
        while i <= n:
            # Split complex eigenvalues into real components
            if d[i].imag != 0:
                T[:, i] = T[:, i].real
                T[:, i+1] = T[:, i+1].imag
                i = i + 2
            else:
                i = i + 1

        return ss.coordinate_transform(invT=T, T=solve(T, np.eye(n)))
    
@dataclass(slots=True)
class BalancedTruncation:
    """
    r: order of the returned reduced-order model

    reduction_method
        - "truncate": Classic balanced truncation, eliminating states with
            small singular values (greater accuracy in high frequency region).
        - "singular perturbation": Balanced truncation, using singular perturbation
            to eliminate fast dynamics (greater accuracy in low frequency region).
    """
    r: int 
    reduction_method: Literal["truncate", "singular perturbation"] = "truncate"

    # TODO: Add option to remove unstable modes?

    def reduce(self, sys:LinearSystem):
        
        if (sys.controllability_gram is None) or (sys.observability_gram is None):
            # TODO: Compute with controls toolbox
            raise KeyError("One or both Gammian's have not been computed")

        R = cholesky(sys.controllability_gram, lower=True)
        L = cholesky(sys.observability_gram, lower=False)

        U, sigma, Vh = svd(L @ R)
        V = Vh.T

        match self.reduction_method:
            case "truncate":
                U_r = U[:, :self.r]
                S_r = np.diag(sigma[:self.r]**(-0.5))
                V_r = V[:, self.r]

                # Reduced similarity transformation matrices (T_r is not square)
                T_r = R @ V_r @ S_r
                invT_r = S_r @ U_r.T @ L

                ss_r = sys.state_space.coordinate_transform(T=T_r, invT=invT_r)

                # 
            case "singular perturbation":
                S = np.diag(sigma**(-0.5))

                # Full similarity transformation matrices (T is square)
                T = R @ V @ S
                invT = S @ U.T @ L

                # Transform to balanced 
                ss = sys.state_space.coordinate_transform(T=T, invT=invT)
                ss_r = singular_perturbation(ss=ss, r=self.r)

        return ss_r

class IRKA:
    pass