# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
import os
from scipy.linalg import block_diag
from more_itertools import transpose
import itertools
import polars as pl


# ------------------
# Import sting code
# ------------------
from sting.system.core import System
import sting.system.selections as sl
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel, modal_analisis
from sting.utils.graph_matrices import get_ccm_matrices, build_ccm_permutation
from sting.modules.power_flow.utils import ACPowerFlowSolution

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
    ccm_matrices: list[np.ndarray] = field(init=False)
    model: StateSpaceModel = field(init=False)
    output_directory: str = None

    def __post_init__(self):
        self.set_output_folder()
        self.system.clean_up()
        self.get_components()
        self.load_ac_power_flow_solution()
        self.construct_components_ssm()
        self.get_ccm_matrices()

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

        generator_dispatch = pl.read_csv(os.path.join(directory, 'generator_dispatch.csv'),
                                         schema_overrides={ 'id': pl.Int64,
                                                            'type': pl.String,
                                                            'timepoint': pl.String,
                                                            'generator': pl.String, 
                                                            'active_power_MW': pl.Float64, 
                                                            'reactive_power_MVAR': pl.Float64})
        bus_voltage = pl.read_csv(os.path.join(directory, 'bus_voltage.csv'),
                                  schema_overrides={ 'id': pl.Int64,
                                                     'timepoint': pl.String,
                                                     'bus': pl.String, 
                                                     'voltage_magnitude_pu': pl.Float64, 
                                                     'voltage_angle_deg': pl.Float64})

        active_generator_dispatch = dict( zip(generator_dispatch.select(['id', 'timepoint', 'type']).iter_rows(), generator_dispatch['active_power_MW']) )
        reactive_generator_dispatch = dict( zip(generator_dispatch.select(['id', 'timepoint', 'type']).iter_rows(), generator_dispatch['reactive_power_MVAR']) )
        bus_voltage_magnitude = dict( zip(bus_voltage.select(['id', 'timepoint']).iter_rows(), bus_voltage['voltage_magnitude_pu']) )
        bus_voltage_angle = dict( zip(bus_voltage.select(['id', 'timepoint']).iter_rows(), bus_voltage['voltage_angle_deg']) )
        
        solution = ACPowerFlowSolution(generator_active_dispatch=active_generator_dispatch,
                                          generator_reactive_dispatch=reactive_generator_dispatch,
                                          bus_voltage_magnitude=bus_voltage_magnitude,
                                          bus_voltage_angle=bus_voltage_angle)

        if timepoint is None:
            t = self.system.tp[0]
        
        self.apply("load_ac_power_flow_solution", t.name, solution)

    def get_components(self):
        """
        Get components that qualified for building the system-scale small-signal model.
        Not all components in system, e.g., bus, line_pi, etc., participate in small-signal modeling.         
        """

        components = []
        for component in self.system:
            if (    hasattr(component, "load_ac_power_flow_solution") 
                and hasattr(component, "_calculate_emt_initial_conditions") 
                and hasattr(component, "_build_small_signal_model")
                ):
                components.append(ComponentSSM(type = component.type, id = component.id))
        
        self.components = components


    def apply(self, method: str, *args):
        """
        Apply a method to the components for small-signal modeling.
        """
        for c in self.components:
               component = getattr(self.system, c.type)[c.id]
               getattr(component, method)(*args)

    def get_ccm_matrices(self):
        """
        Get the CCM matrices in dq frame for the small-signal modeling.
        """
        
        F, G, H, L = get_ccm_matrices(self.system, attribute="ssm", dimI=2)

        T = build_ccm_permutation(self.system)
        T = block_diag(T, np.eye(F.shape[0] - T.shape[0]))

        F = T @ F
        G = T @ G

        self.ccm_matrices = [F, G, H, L]

    def construct_components_ssm(self):
        """
        Create each small-signal model of each component
        """
        self.apply("_calculate_emt_initial_conditions")
        self.apply("_build_small_signal_model")

    def construct_system_ssm(self):
        """
        Return a state-space model of all interconnected components
        """

        # Get components in order of generators, then shunts, then branches
        generators, = self.system.generators.select("ssm")
        shunts, = self.system.shunts.select("ssm")
        branches, = self.system.branches.select("ssm")

        models = itertools.chain(generators, shunts, branches)
     
        # Input of system are device inputs according to defined G matrix
        u = lambda stacked_u: stacked_u[stacked_u.type == "device"]

        # Output of system are all outputs according to defined H matrix
        y = lambda stacked_y: stacked_y
                
        # Then interconnect models
        self.model = StateSpaceModel.from_interconnected(models, self.ccm_matrices, u, y)

        # Print modal analysis
        modal_analisis(self.model.A, show=True)

        # Export small-signal model to CSV files
        self.model.to_csv(self.output_directory)
