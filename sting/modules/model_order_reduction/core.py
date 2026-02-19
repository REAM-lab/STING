# ----------------------
# Import python packages
# ----------------------
from dataclasses import dataclass, field

# ------------------
# Import sting code
# ------------------

from typing import Literal

from scipy.linalg import eig
import numpy as np


from sting.utils.dynamical_systems import StateSpaceModel
from sting.modules.model_order_reduction.utils import singular_perturbation

"""Open-loop reductions"""
@dataclass
class SingularPerturbation:
    r: int 
    basis: Literal["eigen", "none"]

    def reduce(self, sys:StateSpaceModel):
        """Return a reduced-order model."""
        # Perform a coordinate transform to induce timescale separation
        match self.basis:
            case "eigen":
                sys = self._to_eigenbasis(sys)
            case "none":
                pass

        # Compute the ROM
        sys_r = singular_perturbation(sys=sys, r=self.r)
        
        return sys_r
        
    def _to_eigenbasis(sys:StateSpaceModel) -> StateSpaceModel:
        """
        Perform a similarity transform to convert a linear state-space
        into it's modal basis. 
        
        That is, in the returned system, each state (or pair of states)
        corresponds to a model of A. A *real* Jordan form decomposition
        is used to ensure that the returned model is real valued.
        """
        d, V = eig(sys.A)

        # Sort eigenstates from slowest to fastest 
        idx = np.argsort(np.abs(d))
        # Reorder eigenvectors and eigenvalues as needed
        T = V[:, idx]
        d = d[idx]

        # Construct transform T such that J = invT * A * T yields the
        # *real* Jordan form of A
        i = 0
        while i <= len(d):
            # Split complex eigenvalues into real components
            if d[i].imag != 0:
                T[:, i] = T[:, i].real
                T[:, i+1] = T[:, i+1].imag
                i = i + 2
            else:
                i = i + 1

        # TODO: obj.tsys{k} = ss2ss(sys, eye(size(T)) / T);