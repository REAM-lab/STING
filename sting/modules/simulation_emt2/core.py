# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
import itertools
import os
import logging

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
    components: list[str] = None
    variables: VariablesEMT = None
    x_len: int = None
    y_len: int = None
    ud_len: int = None
    x_idx: dict[str, np.ndarray] = None
    xs_idx: dict[str, dict[str, int]] = None
    u_idx: dict[str, np.ndarray] = None
    ud_idx: dict[str, np.ndarray] = None
    us_idx: dict[str, dict[str, int]] = None
    y_idx: dict[str, np.ndarray] = None
    ccm_abc_matrices: list[np.ndarray] = None
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

        self.x_len = len(x)
        self.ud_len = len(ud)
        self.y_len = len(y)

        self.variables = VariablesEMT(x=x, u=u, y=y)

        # Create a dictionary to map component names to their corresponding indices in the x, u, and y variables
        # For example, {'voltage_source_4a_0': [0, 1, 2, 3], 'gfmi_18a_0': [4, 5, 6, 7, 8]}
        self.x_idx = {}
        self.u_idx = {}
        self.ud_idx = {}
        self.y_idx = {}

        for i, component_name in enumerate(x.component):
            self.x_idx.setdefault(component_name, []).append(i)

        for i, component_name in enumerate(u.component):
            self.u_idx.setdefault(component_name, []).append(i)

        for i, component_name in enumerate(ud.component):
            self.ud_idx.setdefault(component_name, []).append(i)

        for i, component_name in enumerate(y.component):
            self.y_idx.setdefault(component_name, []).append(i)

        # Create a dictionary: {'voltage_source_4a_0': {i_bus_a : [1]}, 'gfmi_18a_0': {i_bus_c : [2]}}
        # so we can use xs_idx['voltage_source_4a_0']['i_bus_a']
        self.xs_idx ={}
        for i, xs in enumerate(x):
            component_name = xs.component[0]
            state_name = xs.name[0]
            self.xs_idx.setdefault(component_name, {})[state_name] = i

        # Create a dictionary: {'voltage_source_4a_0': {v_ref_d : [1]}, 'gfmi_18a_0': {p_ref : [2]}}
        self.us_idx ={}
        for i, us in enumerate(u):
            component_name = us.component[0]
            input_name = us.name[0]
            self.us_idx.setdefault(component_name, {})[input_name] = i

    def get_ccm_matrices(self):
        """
        Get the CCM matrices in abc frame for the EMT simulation.
        """
        
        self.ccm_abc_matrices = get_ccm_matrices(self.system, attribute="variables_emt", dimI=3)

    def build_stacked_output(self, x: np.ndarray):

        # Define ystack
        y_stack = np.full(self.y_len, np.nan, dtype=float)

        for c in self.components:
            x_idx = self.x_idx[c.type_ + "_" + str(c.id)]
            x_component = x[x_idx]
            y_idx = self.y_idx[c.type_ + "_" + str(c.id)]
            y_stack[y_idx] = getattr(self.system, c.type_)[c.id].get_output_emt(x_component)

        return y_stack

    def build_device_input(self, u_signals: dict, x: np.ndarray, t: float):
        """
        Get the device input signals for the EMT simulation.
        """

        #ud = np.full(self.ud_len, np.nan, dtype=float)
        ud = np.zeros(self.ud_len, dtype=float)
        d_vars = self.variables.u[self.variables.u.type == "device"]

        for component in u_signals:
            # component = 'gfmi_18a_0'
            for input in u_signals[component]:
                # input = 'v_ref_d'
                ud_idx = self.us_idx[component][input]
                ud[ud_idx] = u_signals[component][input](t)
        ud = ud + d_vars.init
        return ud

    def build_state_derivative(self, x: np.ndarray, ustack: np.ndarray):
        """
        Get the state derivative for the EMT simulation.
        """

        dx_dt = np.full(self.x_len, np.nan, dtype=float)

        for c in self.components:
            x_idx = self.x_idx[c.type_ + "_" + str(c.id)]
            x_component = x[x_idx]
            u_idx = self.u_idx[c.type_ + "_" + str(c.id)]
            u_component = ustack[u_idx]
            dx_dt[x_idx] = getattr(self.system, c.type_)[c.id].get_derivative_state_emt(x_component, u_component)

        return dx_dt

    def set_value(self, time, numerical_vector, var_type: str):
        """
        Update the value of the EMT variables based on a numerical vector
        """

        for c in self.components:
            component = getattr(self.system, c.type_)[c.id]
            variables = getattr(component, "variables_emt")
            x_idx = self.x_idx[c.type_ + "_" + str(c.id)]
            value = numerical_vector[x_idx]

            var_component = getattr(variables, var_type)
            setattr(var_component, "value", value)
            setattr(var_component, "time", time)

    @timeit
    def sim(self, t_max, inputs, settings={'dense_output': True, 'method': 'Radau', 'max_step': 0.001}, components_to_plot=None):
        """
        Run the EMT simulation for the system.
        """

        F, G, _, _ = self.ccm_abc_matrices

        def system_step(t, x, u_signals):
            """
            System step for the EMT simulation.
            """

            # Build device input
            ud = self.build_device_input(u_signals, x, t)

            # Build output
            y_stack = self.build_stacked_output(x)

            ustack = F @ y_stack + G @ ud

            # Build state derivative
            dx_dt = self.build_state_derivative(x, ustack)

            return dx_dt

        solution = solve_ivp(system_step, 
                        [0, t_max], # timeperiod 
                        self.variables.x.init, # initial conditions
                        dense_output=settings['dense_output'],  
                        args=(inputs, ),
                        method=settings['method'], 
                        max_step=settings['max_step'])
        
        # Define timepoints that will be used to evaluate the solution of the ODEs
        if settings['dense_output']:
            tps = np.linspace(0, t_max, 500)
            solution = solution.sol(tps)

        # Set the value of the EMT variables based on the solution of the ODEs
        self.set_value(tps, solution, "x")    

        self.write_results_csv(components=components_to_plot)
        self.plot_results(components=components_to_plot)

    def plot_results(self, components = None):
        """
        Plot EMT simulation results
        """

        if components is None:
            components = self.components

        logger.info(f" - Plotting EMT simulation results in {self.output_directory}")

        for c in components:
            component: Component = getattr(self.system, c.type_)[c.id]
            results: DynamicalVariables = getattr(component, "plot_results_emt")()
            results.to_plotly(figure_filepath=os.path.join(self.output_directory, f"{c.type_}_{c.id}.html"))
    
    def write_results_csv(self, components = None):
        """
        Write EMT simulation results to output directory.
        """

        if components is None:
            components = self.components

        logger.info(f" - Writing EMT simulation results in {self.output_directory}")

        for c in components:
            component: Component = getattr(self.system, c.type_)[c.id]
            results: DynamicalVariables = getattr(component, "plot_results_emt")()
            results.to_timeseries(csv_filepath=os.path.join(self.output_directory, f"{c.type_}_{c.id}.csv"))

