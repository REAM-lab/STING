from dataclasses import dataclass
from typing import Literal

import numpy as np

from sting.modules.model_order_reduction.utils import get_jordan_real_transform, singular_perturbation
from sting.utils.dynamical_systems import StateSpaceModel


@dataclass(slots=True)
class SingularPerturbation:
    r: int 
    basis: Literal["eigen", "none"] = "eigen"

    def reduce(self, sys:StateSpaceModel):
        """Return a reduced-order model."""
        # Perform a coordinate transform to induce timescale separation
        match self.basis:
            case "eigen":
                T, invT = get_jordan_real_transform(sys.A)
                ss = sys.coordinate_transform(T=T, invT=invT)

            case "none":
                I = np.eye(sys.A.size[0])
                T, invT = I, I
                ss = sys

        return singular_perturbation(ss=ss, r=self.r)