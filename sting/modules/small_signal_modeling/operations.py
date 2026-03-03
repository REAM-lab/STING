# -----------------------
# Import python packages
# -----------------------
import logging
from dataclasses import dataclass, field
from itertools import groupby
import copy

import numpy as np
from scipy.linalg import block_diag
# -----------------------
# Import sting code
# -----------------------
from sting.system.core import System
from sting.system.component import Component
from sting.reduced_order_model.linear_rom import LinearROM
from sting.modules.small_signal_modeling.core import SmallSignalModel, ComponentSSM
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables


# Set up logger
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GroupBy:
    """Class to perform operations on small-signal models, such as grouping by zones"""

    model: SmallSignalModel
    by: str
    subsystems: dict[str, SmallSignalModel] = field(init=False, default_factory=dict)

    def __post_init__(self):

        # Copy the full SSM (technically we don't need to copy the system)
        self.model = copy.deepcopy(self.model)

        # Sort all the components re-ordering the interconnection matrices
        self.model.sort_components(by=self.by)

        # Get all of the system's components
        components:list[Component] = [getattr(self.model.system, c.type)[c.id] for c in self.model.components]

        for key, subsystem in groupby(components, key=lambda c: getattr(c, self.by)):
            # For each collection of components in the same zone, create a new SSM
            #system = System(case_directory=self.model.system.case_directory)
            components, models  = [], []

            for component in subsystem:
                #system.add(component)
                components.append(ComponentSSM(component.type_, component.id))
                models.append(component.ssm)
            
            self.subsystems[key] = SmallSignalModel(system=self.model.system, components=components, model=StateSpaceModel.from_stacked(models), post_init=False)


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
        # Number of subsystems 
        s = len(self.subsystems)
        # Create a new system to which we will add subsystem level models 
        new_system = System(case_directory=self.model.system.case_directory)
        new_components = []

        # State-space models and interconnection matrices
        (F,G,H,L) = self.model.ccm_matrices  
        # For each subsystem create a range to index u_stack and y_stack in F, G, and H
        y_stack = np.cumsum([0] + [len(ssm.model.y) for ssm in self.subsystems.values()])
        y = [range(y_stack[i-1], y_stack[i]) for i in range(1, s+1)]
        u_stack = np.cumsum([0] + [len(ssm.model.u) for ssm in self.subsystems.values()])
        u = [range(u_stack[i-1], u_stack[i]) for i in range(1, s+1)]

        (F,G,H,L) = self.model.ccm_matrices

        # Select the block diagonal elements of F corresponding to each subsystem
        diagF = [F[u[i], :][:, y[i]] for i in range(s)]
        Z = F - block_diag(*diagF)
        # Compute the inter-subsystem connection matrix
        m, p = F.shape
        # Set of indices for which there are either device level inputs/outputs 
        # or inter-subsystem level inputs/outputs
        w = np.unique(np.nonzero(np.hstack((G, Z)))[0])
        v = np.unique(np.nonzero(np.vstack((H, Z)))[1])

        # Define selection matrices Phi and Psi (from Lemma 1)
        X = np.zeros((len(w), m)) 
        X[range(len(w)), w] = 1

        Y = np.zeros((p, len(v)))
        Y[v, range(len(v))] = 1

        for i, (key, sub_model) in enumerate(self.subsystems.items()):
            # If component is not assigned to a zone, directly transfer it 
            # to the new system (and do not interconnect components).
            if key is None:
                for c in sub_model.components:
                    id = new_system.add(getattr(self.model.system, c.type)[c.id])
                    new_components.append(ComponentSSM(c.type, id=id))

                    _u, _y = diagF[i].shape

                    sub_model.ccm_matrices = (diagF[i]*0, np.eye(_u), np.eye(_y), np.zeros((_y, _u)))
                continue

            # Matrices used to connect components within each subsystem
            X_i = X[[j for j, k in enumerate(w) if k in u[i]], :][:, u[i]]
            Y_i = Y[:, [j for j, k in enumerate(v) if k in y[i]]][y[i], :]

            # Connect all components within the subsystem
            sub_model.ccm_matrices = (diagF[i], X_i.T, Y_i.T, np.zeros((Y_i.shape[1], X_i.shape[0])))
            # State-space models for each component
            models:list[StateSpaceModel] = sub_model.get_component_attribute("ssm")
            # Define subsystem level inputs and outputs
            inputs = sum([ssm.u for ssm in models], DynamicalVariables(name=[]))
            inputs = inputs[[j for j, k in enumerate(u[i]) if k in w]] # Only select inputs at device-level or from other subsystems
            outputs = sum([ssm.y for ssm in models], DynamicalVariables(name=[]))
            # Interconnect each subsystem
            subsystem_ssm = StateSpaceModel.from_interconnected(models, sub_model.ccm_matrices, y=outputs, u=inputs)
            
            # Add each subsystem level model to the new system and components
            linear_system = LinearROM(ssm=subsystem_ssm)
            new_system.add(linear_system)
            new_components.append(ComponentSSM(linear_system.type_, linear_system.id))

        # Define a new SSM
        new_ssm = SmallSignalModel(
            system=new_system, 
            components=new_components, 
            output_directory=self.model.output_directory,
            post_init=False)
        
        # Update the system-level interconnection matrices
        diagF, diagG, diagH, _ = zip(*[s.ccm_matrices for s in self.subsystems.values()])
        W = block_diag(*diagF)
        X = block_diag(*diagG).T
        Y = block_diag(*diagH).T
        
        F = X @ (F - W) @ Y
        G = X @ G
        H = H @ Y

        new_ssm.ccm_matrices = (F,G,H,L)
        
        return new_ssm
