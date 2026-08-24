from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import cholesky, solve_continuous_lyapunov, svd

from sting.modules.model_order_reduction.utils import singular_perturbation, get_balancing_transform
from sting.reduced_order_model.linear_subsystem import LinearSubsystem


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
    method: Literal["truncate", "singular perturbation"] = "truncate"

    def reduce(self, sys:LinearSubsystem):
        # Unpack state-space matrices
        A,B,C,D = sys.full_order_model.data

        # Compute the gramians of the subsystem
        P = solve_continuous_lyapunov(A, -B@B.T)
        Q = solve_continuous_lyapunov(A.T, -C.T@C)

        if "truncate" == self.method:
            T, invT = get_balancing_transform(P, Q, r=self.r)
            sys_r = sys.full_order_model.coordinate_transform(T=T, invT=invT)

        elif "singular perturbation" == self.method:
            T, invT = get_balancing_transform(P, Q, r=None)
            # Transform to balanced 
            ss_t = sys.full_order_model.coordinate_transform(T=T, invT=invT)
            sys_r = singular_perturbation(ss=ss_t, r=self.r)

        sys.T_l = invT
        sys.T_r = T
        sys.reduced_order_model = sys_r