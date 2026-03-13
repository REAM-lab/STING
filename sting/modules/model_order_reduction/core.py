# ----------------------
# Import python packages
# ----------------------
from typing import Literal
from dataclasses import dataclass, field

from scipy.linalg import eig, solve, cholesky, svd
import numpy as np
# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import StateSpaceModel
from sting.modules.model_order_reduction.utils import singular_perturbation
from sting.reduced_order_model.linear_rom import LinearROM, Reducer

@dataclass(slots=True)
class SingularPerturbation(Reducer):
    r: int 
    basis: Literal["eigen", "none"] = "eigen"

    def reduce(self, sys:LinearROM):
        """Return a reduced-order model."""
        # Perform a coordinate transform to induce timescale separation
        match self.basis:
            case "eigen":
                T, invT = self._to_eigenbasis(sys.full_order_model)
                ss = sys.full_order_model.coordinate_transform(T=T, invT=invT)

            case "none":
                I = np.eye(sys.full_order_model.A.size[0])
                T, invT = I, I
                ss = sys.full_order_model

        # Compute the ROM
        sys.T_l = invT
        sys.T_r = T
        sys.ssm = singular_perturbation(ss=ss, r=self.r)

        
    def _to_eigenbasis(self, ss:StateSpaceModel) -> StateSpaceModel:
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
    
@dataclass(slots=True)
class BalancedTruncation(Reducer):
    """
    r: order of the returned reduced-order model

    reduction_method
        - "truncate": Classic balanced truncation, eliminating states with
            small singular values (greater accuracy in high frequency region).
        - "singular perturbation": Balanced truncation, using singular perturbation
            to eliminate fast dynamics (greater accuracy in low frequency region).
    """
    r: int 
    method: Literal["truncate", "singular perturbation"] = "truncate"
    gramian_c: Literal["subsystem", "lyapunov", "structured", "riccati"] = "lyapunov"
    gramian_o: Literal["subsystem", "lyapunov", "structured", "riccati"] = "lyapunov"

    # TODO: Add option to remove unstable modes?

    def reduce(self, sys:LinearROM):
        
        #if (sys.controllability_gram is None) or (sys.observability_gram is None):
        #    # TODO: Compute with controls toolbox
        #    raise KeyError("One or both Gammian's have not been computed")

        R = cholesky(sys.W_c, lower=True)
        L = cholesky(sys.W_o, lower=False)

        U, sigma, Vh = svd(L @ R)
        V = Vh.T

        match self.method:
            case "truncate":
                U_r = U[:, :self.r]
                S_r = np.diag(sigma[:self.r]**(-0.5))
                V_r = V[:, self.r]

                # Reduced similarity transformation matrices (T_r is not square)
                T = R @ V_r @ S_r
                invT = S_r @ U_r.T @ L

                sys_r = sys.full_order_model.coordinate_transform(T=T, invT=invT)


            case "singular perturbation":
                S = np.diag(sigma**(-0.5))

                # Full similarity transformation matrices (T is square)
                T = R @ V @ S
                invT = S @ U.T @ L

                # Transform to balanced 
                ss = sys.full_order_model.coordinate_transform(T=T, invT=invT)
                sys_r = singular_perturbation(ss=ss, r=self.r)

        sys.T_l = invT
        sys.T_r = T
        sys.ssm = sys_r

class IRKA:
    pass