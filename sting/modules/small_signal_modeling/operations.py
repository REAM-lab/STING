# -----------------------
# Import python packages
# -----------------------
import logging
from dataclasses import dataclass
from itertools import groupby
import copy

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import bsr_array
# -----------------------
# Import sting code
# -----------------------
from sting.modules.small_signal_modeling.core import SmallSignalModel, ComponentSSM
from sting.utils.dynamical_systems import StateSpaceModel
from sting.system.core import System
from sting.utils.data_tools import mat2cell

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

        for key, group in groupby(components, key=lambda c: c.zone):
            # For each collection of components in the same zone create a new SSM
            system = System(case_directory=self.model.system.case_directory)
            components, models  = [], [] # The indexing and SSM of each component

            for component in group:
                system.add(component)
                components.append(ComponentSSM(component.type, component.id))
                models.append(component.ssm)
            
            self.subsystems[key] = SmallSignalModel(
                system=system,
                components=components,
                model=StateSpaceModel.from_stacked(models)
            )

            


    def interconnect(self):
        """
        TODO: Treat empty zones differently
        """

        z = len(self.subsystems)

        # Number of stacked inputs and outputs in each zone
        y_stack = [len(ssm.model.y) for ssm in self.subsystems.values()]
        u_stack = [len(ssm.model.u) for ssm in self.subsystems.values()]

        # Block partition the CCM matrices
        (F,G,H,L) = self.model.ccm_matrices
        # F = mat2cell(F, u_stack, y_stack)
        # G = mat2cell(G, u_stack, None)
        # H = mat2cell(H, None, y_stack)

       

        for i, (zone, model) in enumerate(self.subsystems.items()):
            # j is the set of all zone indices excluding zone i
            j = list(range(z))
            j.remove(i)

            # Inputs in u_B with outputs from another subsystem
            u_B, _ = np.nonzero(np.hstack((G[i], F[i, j])))
            u_B = list(set(u_B))
            # Outputs in y_B with contributions from u_B or y_c
            _, y_B = np.nonzero(np.vstack((H[i], F[j, i])))
            y_B = list(set(y_B))
            
            m, n = len(u_B), len(y_B)
            h, w = F[i, i].shape

            # Matrices used to connect components within each subsystem
            X_i = np.zeros((h, m))
            X_i[u_B, range(m)] = 1

            Y_i = np.zeros((n, w))
            Y_i[range(n), y_B] = 1


            #X_i = full(sparse(1:m, u_B, ones(1,m), m, height(F{i,i})))
            #X_i = bsr_array((np.ones(m), (range(m), u_B)), shape=(m, h)).toarray().T
            
            # Y_i = full(sparse(y_B, 1:n, ones(1,n), width(F{i,i}), n))
            #Y_i = bsr_array((np.ones(n), (y_B, range(n))), shape=(w, n)).toarray().T

            # S_i = [zeros(n,m), Y_i'; 
            #     X_i'      , F{i,i}];
            # s.sys{i} = lft(S_i, s.sys{i});

            model.ccm_matrices = (F[i, i], X_i, Y_i, np.zeros(n, m))
            model.construct_system_ssm(write_csv=False, perform_analysis=False)

            # Connect all components within the subsystem



        X = block_diag(X{:});
        Y = block_diag(Y{:});
        diagF = block_diag(*(F[i,i] for i in range(z)))

        c.F = X*(c.F-diagF)*Y;
        c.G = X*c.G;
        c.H = c.H*Y;
        
        # Create an empty system

        # For each group in groups
        # stack all SSMs if key is not "" and create a subsystem/zone component
            # remove all components from the system
            # add those components to the zone model?
        # if key is ""
        # add each of those indivdual components to the system