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
from sting.utils.dynamical_systems import StateSpaceModel

@dataclass(slots=True)
class InterconnectedBalancedTruncation:
    r: dict[str:int] 
    method: Literal["truncate", "singular perturbation"] = "truncate"

    def reduce(self, small_signal_model:SmallSignalModel):

        ssm = copy.deepcopy(small_signal_model)
        # Interconnect all components in the same zone
        ssm = ssm.group_by("zone").interconnect()
        # Construct a state-space model of the full-order model (FOM)
        model = StateSpaceModel.from_interconnected(
            components=ssm.get_component_attribute("ssm"), 
            connections=ssm.ccm_matrices, 
            u=lambda u: u[u.type == "device"], 
            y=lambda y: y)
    
       # --------------- #
       # Model reduction #
       # --------------- #
        A,B,C,D = model.data

        # Solve for the system-level gramians
        P = solve_continuous_lyapunov(A, -B@B.T)
        Q = solve_continuous_lyapunov(A.T, -C.T@C)

        idx_start, idx_stop = 0, 0
        # Step over each component with a to determine the
        # indices to select from P and Q
        for c in ssm.components:
            component = getattr(ssm.system, c.type)[c.id]
            n = component.ssm.A.shape[0]
            idx_stop += n

            # If the current component is a subsystem try to reduce it
            if c.type == "linear_subsystems":
                sys:LinearSubsystem = component
                r_i = self.r.get(sys.name)
                # Check for undefined reduction order
                if r_i == None:
                    continue
                # Index out the block diagonal matrices of P and Q
                P_i = P[idx_start:idx_stop, idx_start:idx_stop]
                Q_i = Q[idx_start:idx_stop, idx_start:idx_stop]
                
                if "truncate" == self.method:
                    T, invT = get_balancing_transform(P_i, Q_i, r=r_i)
                    sys_r = sys.full_order_model.coordinate_transform(T=T, invT=invT)
                
                elif "singular perturbation" == self.method:
                    T, invT = get_balancing_transform(P_i, Q_i, r=None)
                    # Transform to balanced 
                    ss_t = sys.full_order_model.coordinate_transform(T=T, invT=invT)
                    sys_r = singular_perturbation(ss=ss_t, r=r_i)

                

                # Update the subsystems transform matrices and ROM
                sys.T_l = invT
                sys.T_r = T
                sys.reduced_order_model = sys_r

                # Flip the `ssm` attribute to the reduced order model
                sys.set_using("reduced_order_model")

            # Increment
            idx_start += n

        # Construct the system-level reduced order model
        ssm.model = StateSpaceModel.from_interconnected(
            components=ssm.get_component_attribute("ssm"), 
            connections=ssm.ccm_matrices, 
            u=lambda u: u[u.type == "device"], 
            y=lambda y: y)

        return ssm