# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from collections import namedtuple
from collections.abc import Iterable
from typing import NamedTuple
import os
from scipy.linalg import block_diag
import itertools
import polars as pl
from itertools import groupby
import copy

from sting.generator.linear_system import LinearSystem
#from sting.modules.small_signal_modeling.core import SmallSignalModel, ComponentSSM
#from sting.utils.dynamical_systems import StateSpaceModel
#from sting.system.core import System
from sting.utils.data_tools import mat2cell, cell2mat
# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.system.component import Component
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel, modal_analisis
from sting.modules.small_signal_modeling.utils import get_ccm_matrices, build_ccm_permutation
from sting.modules.power_flow.utils import ACPowerFlowSolution
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
    A component of the system that participates in small-signal modeling.

    #### Attributes:
    - type: `str`
            inf_src, se_rl, pa_rc, ... etc. 
    - idx: `int`
            Index of the component in its corresponding list in the system.
    """
    type: str
    id: int

class ConnectionMatrices(NamedTuple):
    """
    Component connection matrices
    Using a NamedTuple to avoid accessing each element by it's index in a list
    """
    F: np.ndarray
    G: np.ndarray
    H: np.ndarray
    L: np.ndarray

# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class SmallSignalModel:
    system: System 
    components: list[ComponentSSM] = None
    model: StateSpaceModel = None
    # Component connection matrices
    F: np.ndarray = None
    G: np.ndarray = None
    L: np.ndarray = None
    H: np.ndarray = None
    output_directory: str = None
    post_init: bool = True

    def __post_init__(self):
        if self.post_init:
            self.set_output_folder()
            #self.system.clean_up()
            self.load_components()
            self.load_ac_power_flow_solution()
            self.construct_components_ssm()
            self.construct_ccm_matrices()

    def set_output_folder(self):
        """
        Set up the output folder for storing results.
        """
        if self.output_directory is None:
            self.output_directory = os.path.join(self.system.case_directory, "outputs", "small_signal_model")
        os.makedirs(self.output_directory, exist_ok=True)

    def load_ac_power_flow_solution(self, timepoint = None, directory: str = None):
        """
        Upload the solution of the optimization model back to the system object.
        """
        if directory is None:
            directory = os.path.join(self.system.case_directory, "outputs", "ac_power_flow")

        generator_dispatch = pl.read_csv(
            source=os.path.join(directory, 'generator_dispatch.csv'),
            schema_overrides={
                'id': pl.Int64,
                'type': pl.String,
                'timepoint': pl.String,
                'generator': pl.String, 
                'active_power_MW': pl.Float64, 
                'reactive_power_MVAR': pl.Float64
            }
        )
        bus_voltage = pl.read_csv(
            source=os.path.join(directory, 'bus_voltage.csv'),
            schema_overrides={
                'id': pl.Int64,
                'timepoint': pl.String,
                'bus': pl.String, 
                'voltage_magnitude_pu': pl.Float64, 
                'voltage_angle_deg': pl.Float64
            }
        )

        generator_keys = list(generator_dispatch.select(['id', 'timepoint', 'type']).iter_rows())
        active_generator_dispatch = dict( zip(generator_keys, generator_dispatch['active_power_MW']) )
        reactive_generator_dispatch = dict( zip(generator_keys, generator_dispatch['reactive_power_MVAR']) )

        bus_keys = list(bus_voltage.select(['id', 'timepoint']).iter_rows())
        bus_voltage_magnitude = dict( zip(bus_keys, bus_voltage['voltage_magnitude_pu']) )
        bus_voltage_angle = dict( zip(bus_keys, bus_voltage['voltage_angle_deg']) )
        
        solution = ACPowerFlowSolution(
            generator_active_dispatch=active_generator_dispatch,
            generator_reactive_dispatch=reactive_generator_dispatch,
            bus_voltage_magnitude=bus_voltage_magnitude,
            bus_voltage_angle=bus_voltage_angle)

        if timepoint is None:
            t = self.system.timepoints[0]
        
        self.apply("load_ac_power_flow_solution", t.name, solution)

    def load_components(self):
        """
        Get components that qualified for building the system-scale small-signal model. 
        Components should be sorted in the order in which the interconnection 
        matrices are constructed (i.e., generators, shunts, branches).        
        """
        if self.components is None:
            ssm_components:Iterable[Component] = itertools.chain(self.system.gens, self.system.shunts, self.system.branches)
            self.components = [ComponentSSM(type=c.type_, id=c.id) for c in ssm_components]

    def construct_ccm_matrices(self):
        """
        Initialize the CCM matrices in dq frame for the small-signal modeling.
        """
        if all([X is None for X in self.ccm_matrices]):
            self.F, self.G, self.H, self.L = get_ccm_matrices(self.system, attribute="ssm", dimI=2)
            # Permute the F and G 
            T = build_ccm_permutation(self.system)
            T = block_diag(T, np.eye(self.F.shape[0] - T.shape[0]))

            self.F = T @ self.F
            self.G = T @ self.G

    def construct_components_ssm(self):
        """
        Create each small-signal model of each component
        """
        self.apply("_calculate_emt_initial_conditions")
        self.apply("_build_small_signal_model")

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
            self.model.to_csv(self.output_directory)

    def sort_components(self, by):
        """
        Sort the components in the small-signal model according
        to one of their attributes. Implicitly this will re-order
        the inputs, outputs, and states of the resulting SSM.
        """
        # Sort components using the attribute "by" as a sorting key
        zones = self.get_component_attribute(by)
        # Sorted ids for every component
        ids, _ = zip(*sorted(zip(range(len(zones)), zones), key=lambda x: (1, x[1]) if (x[1] is not None) else (0, "")))

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

    def group_by(self, by):
        return GroupBy(self, by=by)

    # --------------
    # Helpers
    # --------------
    @property
    def ccm_matrices(self) -> ConnectionMatrices:
        return ConnectionMatrices(self.F, self.G, self.H, self.L)
    
    @ccm_matrices.setter
    def ccm_matrices(self, value):
        if len(value) != 4:
            raise ValueError("Exactly four connection matrices must be provided.")
        
        self.F, self.G, self.H, self.L = value

    def get_component_attribute(self, attribute):
        """Return a list of the specified attribute for every SSM component."""
        return [getattr(getattr(self.system, c.type)[c.id], attribute) for c in self.components]

    def apply(self, method: str, *args):
        """Execute a method of all SSM components."""
        for c in self.components:
            component = getattr(self.system, c.type)[c.id]
            getattr(component, method)(*args) 


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

        for key, subsystem in groupby(components, key=lambda c: c.zone):
            # For each collection of components in the same zone, create a new SSM
            system = System(case_directory=self.model.system.case_directory)
            components, models  = [], []

            for component in subsystem:
                system.add(component)
                components.append(ComponentSSM(component.type_, component.id))
                models.append(component.ssm)
            
            self.subsystems[key] = SmallSignalModel(system=system, components=components, model=StateSpaceModel.from_stacked(models), post_init=False)


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
        s = len(self.subsystems)

        new_system = System(case_directory=self.model.system.case_directory)
        new_components = []

        # State-space models and interconnection matrices
        (F,G,H,L) = self.model.ccm_matrices  
        # Number of stacked inputs and outputs in each zone
        y_stack = np.cumsum([0] + [len(ssm.model.y) for ssm in self.subsystems.values()])
        y = [range(y_stack[i-1], y_stack[i]) for i in range(1, s+1)]
        u_stack = np.cumsum([0] + [len(ssm.model.u) for ssm in self.subsystems.values()])
        u = [range(u_stack[i-1], u_stack[i]) for i in range(1, s+1)]

        # Block partition the CCM matrices
        (F,G,H,L) = self.model.ccm_matrices

        diagF = [F[u[i], :][:, y[i]] for i in range(s)]
        Z = F - block_diag(*diagF)
        m, p = F.shape

        w = np.unique(np.nonzero(np.hstack((G, Z)))[0])
        v = np.unique(np.nonzero(np.vstack((H, Z)))[1])

        X = np.zeros((len(w), m)) 
        X[range(len(w)), w] = 1

        Y = np.zeros((p, len(v)))
        Y[v, range(len(v))] = 1

        for i, (key, sub_model) in enumerate(self.subsystems.items()):
            # If component is not assigned to a zone, directly transfer it 
            # to the new system (and do not interconnect components).
            if key is None:
                for c in sub_model.components:
                    new_system.add(getattr(self.model.system, c.type)[c.id])
                    new_components.append(c)

                continue

            # Matrices used to connect components within each subsystem
            X_i = X[[j for j, k in enumerate(w) if k in u[i]], :][:, u[i]]
            Y_i = Y[:, [j for j, k in enumerate(v) if k in y[i]]][y[i], :]

            # Connect all components within the subsystem
            sub_model.ccm_matrices = (diagF[i], X_i.T, Y_i.T, np.zeros((Y_i.shape[1], X_i.shape[0])))
            #TODO: Make sure inputs and outputs are being defined correctly in this step
            # sub_model.construct_system_ssm(write_csv=False, perform_analysis=False)
            models = sub_model.get_component_attribute("ssm")
            inputs = sum([ssm.u for ssm in models], DynamicalVariables(name=[]))
            outputs = sum([ssm.y for ssm in models], DynamicalVariables(name=[]))
            #u = sum(ssm.u, DynamicalVariables(name=[]) for)
            ssm = StateSpaceModel.from_interconnected(models, sub_model.ccm_matrices, y=outputs, u=inputs[[j for j, k in enumerate(w) if k in u[i]]])


            # Add a zone level model to the system
            linear_system = LinearSystem(ssm=ssm)
            new_system.add(linear_system) 
            new_components.append(ComponentSSM(linear_system.type_, linear_system.id))

        # Update the system-level interconnection matrices       
        F = X @ Z @ Y
        G = X @ G
        H = H @ Y

        return SmallSignalModel(
            system=new_system, 
            components=new_components, 
            F=F, G=G, H=H, L=L, 
            output_directory=self.model.output_directory,
            post_init=False)