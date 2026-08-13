import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from sting.modules.model_order_reduction.utils import (
    get_balancing_transform,
    singular_perturbation,
)
from sting.modules.small_signal_modeling.core import SmallSignalModel
from sting.reduced_order_model.linear_subsystem import LinearSubsystem


@dataclass(slots=True)
class InterconnectedBalancedTruncation:
    r: dict[str:int] 
    method: Literal["truncate", "singular perturbation"] = "truncate"

    def reduce(self, small_signal_model:SmallSignalModel):
        small_signal_model = copy.deepcopy(small_signal_model)


        A,B,C,D = small_signal_model.model.data

        # Solve for the system-level gramians
        P = solve_continuous_lyapunov(A, -B@B.T)
        Q = solve_continuous_lyapunov(A.T, -C.T@C)

        idx_start, idx_stop = 0, 0
        # Step over each component with a to determine the
        # indices to select from P and Q
        for c in small_signal_model.components:
            component = getattr(small_signal_model.system, c.type)[c.id]
            n = component.ssm.A.shape[0]
            idx_stop += n

            if c.type == "linear_subsystem":
                sys:LinearSubsystem = component
                r_i = self.r[sys.zone]
                if r_i == None:
                    continue
                P_i = P[idx_start:idx_stop, idx_start:idx_stop]
                Q_i = Q[idx_start:idx_stop, idx_start:idx_stop]
                
                if "truncate" == self.method:
                    T, invT = get_balancing_transform(P, Q, r=r_i)
                    sys_r = sys.full_order_model.coordinate_transform(T=T, invT=invT)
                
                elif "singular perturbation" == self.method:
                    T, invT = get_balancing_transform(P, Q, r=None)
                    # Transform to balanced 
                    ss_t = sys.full_order_model.coordinate_transform(T=T, invT=invT)
                    sys_r = singular_perturbation(ss=ss_t, r=r_i)

            sys.T_l = invT
            sys.T_r = T
            sys.reduced_order_model = sys_r
            
            idx_start += n

        small_signal_model.construct_system_ssm(self, write_csv=False, perform_analysis=False)

        return small_signal_model