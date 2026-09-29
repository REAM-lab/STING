import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from sting.modules.model_order_reduction.utils import (
    get_balancing_transform,
    singular_perturbation,
)
from sting.modules.model_order_reduction.core import ModelReducer


@dataclass(slots=True)
class InterconnectedBalancedTruncation:
    r: dict[str:int] 
    method: Literal["truncate", "singular perturbation"] = "truncate"

    def reduce(self, model_reducer:ModelReducer):

        if len(model_reducer.zones) > len(set(model_reducer.zones)):
            raise TypeError("Zonal models have not been constructed, please call " \
            "`create_zonal_models` prior to reduction.")

        system_model = model_reducer.interconnect()
        zones = model_reducer.zones
       # --------------- #
       # Model reduction #
       # --------------- #
        reduced_models = []
        A,B,C,D = system_model.data

        # Solve for the system-level gramians
        P = solve_continuous_lyapunov(A, -B@B.T)
        Q = solve_continuous_lyapunov(A.T, -C.T@C)

        idx_start, idx_stop = 0, 0
        # Step over each component with a to determine the
        # indices to select from P and Q
        for zone, zonal_model in zip(zones, model_reducer.models):
            n = zonal_model.A.shape[0]
            idx_stop += n

            # If the current component is in the reduction dict, reduce it
            if zone in self.r:
                # Target reduction order of zone i
                r_i = self.r[zone]

                # Index out the block diagonal matrices of P and Q
                P_i = P[idx_start:idx_stop, idx_start:idx_stop]
                Q_i = Q[idx_start:idx_stop, idx_start:idx_stop]
                
                if "truncate" == self.method:
                    T, invT = get_balancing_transform(P_i, Q_i, r=r_i)
                    sys_r = zonal_model.coordinate_transform(T=T, invT=invT)
                
                elif "singular perturbation" == self.method:
                    T, invT = get_balancing_transform(P_i, Q_i, r=None)
                    # Transform to balanced 
                    ss_t = zonal_model.coordinate_transform(T=T, invT=invT)
                    sys_r = singular_perturbation(ss=ss_t, r=r_i)

                reduced_models.append(sys_r)

            else:
                reduced_models.append(zonal_model)

            # Increment
            idx_start += n

        # Return reduced order model
        F,G,H,L = model_reducer.connections
        return ModelReducer(reduced_models, zones, F, G, H, L, model_reducer.model_class)        