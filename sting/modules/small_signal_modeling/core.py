# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
import os
from scipy.linalg import block_diag
import itertools
import polars as pl


# ------------------
# Import sting code
# ------------------
from sting.system.core import System
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

# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class SmallSignalModel:
    system: System 
    components: list[ComponentSSM] = field(init=False)
    model: StateSpaceModel = field(init=False)
    # Component connection matrices
    F: np.ndarray = field(init=False)
    G: np.ndarray = field(init=False)
    L: np.ndarray = field(init=False)
    H: np.ndarray = field(init=False)
    output_directory: str = None

    def __post_init__(self):
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
        Not all components in system, e.g., bus, line_pi, etc., participate in small-signal modeling.         
        """

        # Get components that qualified for building the system-scale small-signal model. 
        # Components should be sorted in the order in which the interconnection 
        # matrices are constructed (i.e., generators, shunts, branches).
        ssm_components = itertools.chain(self.system.generators, self.system.shunts, self.system.branches)
        self.components = [ComponentSSM(type=c.type, id=c.id) for c in ssm_components]

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
        """Execute a method of all SSM components."""
        for c in self.components:
            component = getattr(self.system, c.type)[c.id]
            getattr(component, method)(*args) 
