from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import cholesky, solve_continuous_lyapunov, svd

from sting.modules.model_order_reduction.utils import (
    get_balancing_transform,
    singular_perturbation,
    controllability_cholesky,
    observability_cholesky,
)
from sting.reduced_order_model.linear_subsystem import LinearSubsystem


@dataclass(slots=True)
class BalancedTruncation:
    """
    Parameters
    ----------
    r: Order of the returned reduced-order model
    reduction_method
        - "truncate": Classic balanced truncation, eliminating states with
            small singular values (greater accuracy in high frequency region).
        - "singular perturbation": Balanced truncation, using singular perturbation
            to eliminate fast dynamics (greater accuracy in low frequency region),
            also known as "matchDC" in MATLAB.
    library: Backend library used for model reduction. We include a `scipy` implementation
        however `SLICOT` can be more numerically robust across machines.
    tol: Tolerance at which to treat a singular value as zero.
    """
    r: int 
    method: Literal["truncate", "singular perturbation"] = "truncate"
    library: Literal["slycot", "slycot-sqrt", "scipy"] = "slycot"
    tol: float = 0

    def reduce(self, sys:LinearSubsystem):
        # Unpack state-space matrices
        A,B,C,D = sys.full_order_model.data

        if self.library == "slycot":
            # Use SLICOT to compute balance truncation directly
            self._slycot_reduce(sys)
            return 

        elif self.library == "scipy":
            # Compute the gramians of the subsystem using scipy
            P = solve_continuous_lyapunov(A, -B@B.T)
            Q = solve_continuous_lyapunov(A.T, -C.T@C)
            R = None
            L = None

        elif self.library == "slycot-sqrt":
            # Compute the cholesky factorizations of P and Q directly with SLICOT
            P = None
            Q = None
            R, _ = controllability_cholesky(A, B, lower=True)
            L, _ = observability_cholesky(A, C, lower=True)

        # Reduction method
        if "truncate" == self.method:
            T, invT = get_balancing_transform(P, Q, r=self.r, R=R, L=L)
            sys_r = sys.full_order_model.coordinate_transform(T=T, invT=invT)

        elif "singular perturbation" == self.method:
            T, invT = get_balancing_transform(P, Q, r=None, R=R, L=L)
            # Transform to balanced 
            ss_t = sys.full_order_model.coordinate_transform(T=T, invT=invT)
            sys_r = singular_perturbation(ss=ss_t, r=self.r)

        # Save the transform matrices
        sys.T_l = invT
        sys.T_r = T
        # Name the new system component ROM
        sys_r.x.component = 'rom'
        # Save the ROM
        sys.reduced_order_model = sys_r


    def _slycot_reduce(self, sys:LinearSubsystem):
        from slycot import ab09ad, ab09nd
        
        from sting.utils.dynamical_systems import (
            DynamicalVariables,
            StateSpaceModel,
        )
        A,B,C,D = sys.full_order_model.data
        n, m = B.shape
        p = C.shape[0]

        if self.method == 'truncate':
            Nr, Ar, Br, Cr, hsv = ab09ad(
                dico='C', job='B', equil='N', n=n, m=m, p=p, 
                A=A, B=B, C=C, nr=self.r, tol=0)
            Dr = D

        elif self.method == "singular perturbation":
            Nr, Ar, Br, Cr, Dr, Ns, hsv = ab09nd(
                dico='C', job='B', equil='N', n=n, m=m, p=p, 
                A=A, B=B, C=C, D=D,
                alpha=0, nr=self.r, tol1=0, tol2=0.0)

        x = DynamicalVariables(name=[f"x{i}" for i in range(self.r)], component='rom', init=np.zeros(self.r))
        sys.reduced_order_model = StateSpaceModel(A=Ar, B=Br,C=Cr,D=Dr, x=x, u=sys.full_order_model.u, y=sys.full_order_model.y)