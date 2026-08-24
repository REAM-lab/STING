from dataclasses import dataclass
from typing import Literal

import numpy as np

from sting.modules.model_order_reduction.utils import get_jordan_real_transform, singular_perturbation
from sting.reduced_order_model.linear_subsystem import LinearSubsystem


@dataclass(slots=True)
class SingularPerturbation:
    r: int 
    basis: Literal["eigen", "none"] = "eigen"

    def reduce(self, sys:LinearSubsystem):
        """Return a reduced-order model."""
        # Perform a coordinate transform to induce timescale separation
        match self.basis:
            case "eigen":
                T, invT = get_jordan_real_transform(sys.full_order_model.A)
                ss = sys.full_order_model.coordinate_transform(T=T, invT=invT)

            case "none":
                I = np.eye(sys.full_order_model.A.size[0])
                T, invT = I, I
                ss = sys.full_order_model

        # Compute the ROM
        sys.T_l = invT
        sys.T_r = T
        sys.reduced_order_model = singular_perturbation(ss=ss, r=self.r)