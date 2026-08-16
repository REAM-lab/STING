# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
import itertools
import os
import logging
from collections import defaultdict

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.system.component import Component
from sting.utils.dynamical_systems import DynamicalVariables
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.component_connections import get_ccm_matrices
from sting.utils.runtime_tools import timeit
from sting.modules.power_flow.utils import load_ac_power_flow_solution

# Set up logging
logger = logging.getLogger(__name__)

# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class SimulationEMT:

    system: System
    components: list[str] = field(init=False)
    component_to_xidx: dict[str, np.ndarray] = field(init=False)
    component_to_uidx: dict[str, np.ndarray] = field(init=False)
    component_to_yidx: dict[str, np.ndarray] = field(init=False)
    ccm_abc_matrices: list[np.ndarray] = field(init=False)
    power_flow_directory: str = None
    output_directory: str = None

    def set_output_folder(self):
        """
        Set up the output folder for storing results.
        """
        if self.output_directory is None:
            self.output_directory = os.path.join(self.system.case_directory, "outputs", "simulation_emt")
        os.makedirs(self.output_directory, exist_ok=True)

    def get_components(self):
        """
        Get components that qualified for building the differential equations for EMT dynamics.
        Not all components in system, e.g., bus, line_pi, etc., participate in EMT simulation.         
        """

        components: list[Component] = []
        for component in self.system:
            if (    
                    hasattr(component, "define_variables_emt")
                and hasattr(component, "get_derivative_state_emt")
                and hasattr(component, "get_output_emt")
                and hasattr(component, "plot_results_emt")
                ):
                components.append(Component(type_ = component.type_, id = component.id))
        
        self.components = components

    def initialize_variables(self):
        """
        Initialize the EMT variables for all components in the system.
        """

        # Get the solution of the AC power flow model
        if self.power_flow_directory is None:
            self.power_flow_directory = os.path.join(self.system.case_directory, "outputs", "ac_power_flow")

        solution = load_ac_power_flow_solution(self.power_flow_directory)

        # Default to the first timepoint if no timepoint is specified
        t = self.system.timepoints[0]

        self.apply("load_ac_power_flow_solution", t.name, solution)

        self.apply("_calculate_emt_initial_conditions")

    def apply(self, method: str, *args):
        """
        Apply a method to the components for EMT simulation.
        """
        for c in self.components:
               component = getattr(self.system, c.type_)[c.id]
               getattr(component, method)(*args)

    def get_variables(self):
        """
        Define EMT variables for all components in the system
        """
        self.apply("define_variables_emt")

        # TODO: filter out components using list of components that are not participating in EMT simulation
        generators, = self.system.ccm_generators.select("variables_emt")
        shunts, = self.system.ccm_shunts.select("variables_emt")
        branches, = self.system.ccm_branches.select("variables_emt")

        variables_emt = itertools.chain(generators, shunts, branches)

        fields = ["x", "u", "y"]
        selection = [[getattr(c, f) for f in fields] for c in variables_emt]
        stack = dict(zip(fields, zip(*selection)))

        x = sum(stack["x"], DynamicalVariables(name=[]))
        y = sum(stack["y"], DynamicalVariables(name=[]))

        u = sum(stack["u"], DynamicalVariables(name=[]))
        ud = u[u.type == "device"]
        ug = u[u.type == "grid"]
        u = ud + ug


        # Create a dictionary to map component names to their corresponding indices in the x, u, and y variables
        # For example, {'voltage_source_4a_0': [0, 1, 2, 3], 'gfmi_18a_0': [4, 5, 6, 7, 8]}
        self.component_to_xidx = {}
        self.component_to_uidx = {}
        self.component_to_yidx = {}

        for i, name in enumerate(x.component):
            self.component_to_xidx.setdefault(name, []).append(i)

        for i, name in enumerate(u.component):
            self.component_to_uidx.setdefault(name, []).append(i)

        for i, name in enumerate(y.component):
            self.component_to_yidx.setdefault(name, []).append(i)

    def get_ccm_matrices(self):
        """
        Get the CCM matrices in abc frame for the EMT simulation.
        """
        
        self.ccm_abc_matrices = get_ccm_matrices(self.system, attribute="variables_emt", dimI=3)

    def get_stacked_output(self, x):

        # Define ystack
        y_stack = np.full(y_len, np.nan, dtype=float)

        for c in self.components:
            y_idx = self.component_to_yidx[c.type_ + "_" + c.id]
            y_stack[y_idx] = getattr(self.system, c.type_)[c.id].get_output_emt(x)

        return y_stack