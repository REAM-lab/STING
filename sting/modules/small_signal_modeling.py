# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
import os
from scipy.linalg import block_diag
import itertools

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel, modal_analisis
from sting.utils.graph_matrices import get_ccm_matrices, build_ccm_permutation
from sting.utils.data_tools import block_permute

# -----------
# Sub-classes
# -----------
class VariablesSSM(NamedTuple):
    """
    All variables in the system for small-signal modeling.
    """
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables


class ComponentSSM(NamedTuple):
    """
    A light-weight wrapper tracking the components of the system 
    that are used in small-signal modeling.

    Attributes:
    - type: `str`
        inf_src, se_rl, pa_rc, ... etc. 
    - idx: `int`
        Index of the component in its corresponding list in the system.
    """
    type: str
    id: int

# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class SmallSignalModel:
    system: System
    components: list[ComponentSSM] = field(init=False)
    # 
    model: StateSpaceModel = field(init=False)
    # Component connection matrices
    F: np.ndarray = field(init=False)
    G: np.ndarray = field(init=False)
    L: np.ndarray = field(init=False)
    H: np.ndarray = field(init=False)

    settings: None

    def __post_init__(self):
        self.system.clean_up()
        
        # Get components that qualified for building the system-scale small-signal model. 
        # Components should be sorted in the order in which the interconnection 
        # matrices are constructed (i.e., generators, shunts, branches).
        ssm_components = itertools.chain(self.system.generators, self.system.shunts, self.system.branches)
        self.components = [ComponentSSM(type=c.type, id=c.id) for c in ssm_components]

        # Create each small-signal model of each component
        self.apply("_calculate_emt_initial_conditions")
        self.apply("_build_small_signal_model")

        # Construct the CCM interconnection matrices
        self.ccm_matrices = self.construct_ccm_matrices()

    
    def construct_ccm_matrices(self):
        """
        Initialize the CCM matrices in dq frame for the small-signal modeling.
        """
        self.F, self.G, self.H, self.L = get_ccm_matrices(self.system, attribute="ssm", dimI=2)
        # Permute the F and G 
        T = build_ccm_permutation(self.system)
        T = block_diag(T, np.eye(self.F.shape[0] - T.shape[0]))

        self.F = T @ self.F
        self.G = T @ self.G


    def construct_system_ssm(self, write_csv=True, perform_analysis=True):
        """
        Return a state-space model of all interconnected components
        """
        # State-space model for each component
        models = self.get_component_attribute("ssm")
     
        # Input of system are device inputs (according to defined G matrix)
        u = lambda u: u[u.type == "device"]
        # Output of system are all outputs (according to defined H matrix)
        y = lambda y: y

        # Then interconnect models
        self.model = StateSpaceModel.from_interconnected(models, self.ccm_matrices, u, y)

        # Print modal analysis
        if perform_analysis:
            modal_analisis(self.model.A, show=True)

        # Export small-signal model to CSV files
        if write_csv:
            file_path = os.path.join(self.system.case_directory, "outputs", "small_signal_model")
            self.model.to_csv(file_path)


    def sort_components_by(self, by="zone"):
        # Sort components using the attribute "by" as a sorting key
        zones = self.get_component_attribute(by)
        # Sorted ids for every component
        ids, _ = zip(*sorted(zip(range(len(zones)), zones)), key=lambda x: x[1])

        # SSMs for each component
        models = self.get_component_attribute("ssm")

        # Total number of inputs/outputs for each component 
        y_stack = [len(ssm.y) for ssm in models]
        u_stack = [len(ssm.u) for ssm in models]

        # Number input/outputs for each component at the system-level.
        # We assume component and system-level outputs are the same.
        y_system = y_stack 
        u_system = [ssm.u.n_device for ssm in models]

        # Permute each component connection matrix to correspond to
        # the sorted components
        self.F = block_permute(self.F, u_stack,  y_stack,  ids)
        self.G = block_permute(self.G, u_stack,  u_system, ids)
        self.H = block_permute(self.H, y_system, y_stack,  ids)
        self.L = block_permute(self.L, y_system, u_system, ids)

        # And sort all the components
        self.components = [self.components[i] for i in ids]


    # --------------
    # Helpers
    # --------------
    @property
    def ccm_matrices(self):
        return (self.F, self.G, self.H, self.L)

    def get_component_attribute(self, attribute):
        """Return a list of the specified attribute for every SSM component."""
        return [getattr(getattr(self.system, c.type)[c.idx], attribute) for c in self.components]


    def apply(self, method: str, *args):
        """Execute a method to all  SSM components."""
        for c in self.components:
            component = getattr(self.system, c.type)[c.id]
            getattr(component, method)(*args)