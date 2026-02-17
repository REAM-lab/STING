# -----------------------
# Import python packages
# -----------------------
import logging
from dataclasses import dataclass
from itertools import groupby, transpose
import copy

import numpy as np
from scipy.linalg import block_diag
# -----------------------
# Import sting code
# -----------------------
from sting.generator.linear_system import LinearSystem
from sting.modules.small_signal_modeling.core import SmallSignalModel, ComponentSSM
from sting.utils.dynamical_systems import StateSpaceModel
from sting.system.core import System
from sting.utils.data_tools import mat2cell, cell2mat

# Set up logger
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SSMGroupBy:
    """Class to perform operations on small-signal models, such as grouping by zones"""

    model: SmallSignalModel
    by: str
    subsystems: dict[str, SmallSignalModel]

    def __post_init__(self):

        # Copy the full SSM (technically we don't need to copy the system)
        self.model = copy.deepcopy(self.model)

        # Sort all the components re-ordering the interconnection matrices
        self.model.sort_components(by=self.by)

        # Get all of the system's components
        components = [getattr(self.model.system, c.type)[c.idx] for c in self.model.components]

        for key, subsystem in groupby(components, key=lambda c: c.zone):
            # For each collection of components in the same zone, create a new SSM
            system = System(case_directory=self.model.system.case_directory)
            components, models  = [], []

            for component in subsystem:
                system.add(component)
                components.append(ComponentSSM(component.type, component.id))
                models.append(component.ssm)
            
            self.subsystems[key] = SmallSignalModel(system=system, components=components, model=StateSpaceModel.from_stacked(models))


    def interconnect(self) -> SmallSignalModel:
        """
        Now we have grouped each subsystem but there are no 
        internal connections within subsystems. Here we remove all inputs and 
        outputs from each subsystem that have no contribution to y_C.
        
        Recall from CCM that
            u_B = F * y_B + G * u_C
            y_C = H * y_B + L * u_C
        
        Specifically, we construct two more maskings for u_B and y_B. These
        correspond to inputs and outputs that have no contribution to u_C or y_C,
        and thus can be eliminated from each subsystem without any effect on the 
        fully interconnected dynamics of G_C(s). We the matrix construct X to 
        mask u_B such that u_B(mask) = X * u_B meaning
            X * u_B = (X*F) * y_B + (X*G) * u_B
        and the matrix Y to mask y_B such that y_B(mask) = Y * y_B meaning
            u_B = (F*Y) * y_B + G * u_C
            y_C = (H*Y) * y_B + L * u_C
        """
        new_system = System(case_directory=self.model.system.case_directory)
        new_components = []

        # State-space models and interconnection matrices
        (F,G,H,L) = self.model.ccm_matrices  
        # Number of stacked inputs and outputs in each zone
        y_stack = [len(ssm.model.y) for ssm in self.subsystems.values()]
        u_stack = [len(ssm.model.u) for ssm in self.subsystems.values()]

        # Block partition the CCM matrices
        (F,G,H,L) = self.model.ccm_matrices
        F_cell = mat2cell(F, u_stack, y_stack)
        G_cell = mat2cell(G, u_stack, None)
        H_cell = mat2cell(H, None, y_stack)

        for i, (key, sub_model) in enumerate(self.subsystems.items()):
            # If component is not assigned to a zone, directly transfer it 
            # to the new system (and do not interconnect components).
            if key == "":
                for c in sub_model.components:
                    new_system.add(getattr(self.model.system, c.type)[c.idx])
                    new_components.append(c)

                continue

            # j is the set of all zone indices excluding zone i
            j = list(range(len(self.subsystems)))
            j.remove(i)

            # Inputs in u_B with outputs from another subsystem
            u_B, _ = np.nonzero(np.hstack((cell2mat(G_cell[i]), cell2mat(F_cell[i, j]))))
            u_B = list(set(u_B))
            # Outputs in y_B with contributions from u_B or y_c
            _, y_B = np.nonzero(np.vstack((cell2mat(H_cell[i]), cell2mat(F_cell[j, i]))))
            y_B = list(set(y_B))
            
            m, n = len(u_B), len(y_B)
            h, w = F[i, i].shape

            # Matrices used to connect components within each subsystem
            X_i = np.zeros((h, m))
            X_i[u_B, range(m)] = 1

            Y_i = np.zeros((n, w))
            Y_i[range(n), y_B] = 1

            # Connect all components within the subsystem
            sub_model.ccm_matrices = (F[i, i], X_i, Y_i, np.zeros(n, m))
            #TODO: Make sure inputs and outputs are being defined correctly in this step
            sub_model.construct_system_ssm(write_csv=False, perform_analysis=False)
            
            # Add a zone level model to the system
            linear_system = LinearSystem(sub_model.ssm)
            new_system.add(linear_system) 
            new_components.append(ComponentSSM(linear_system.type, linear_system.id))

        # Update the system-level interconnection matrices
        diagF, diagG, diagH, _ = transpose([s.ccm_matrices for s in self.subsystems.values()])
        W = block_diag(*diagF)
        X = block_diag(*diagG)
        Y = block_diag(*diagH)
        
        F = X @ (F - W) @ Y
        G = X @ G
        H = H @ Y

        return SmallSignalModel(
            system=new_system, 
            components=new_components, 
            F=F, G=G, H=H, L=L, 
            output_directory=self.model.output_directory)