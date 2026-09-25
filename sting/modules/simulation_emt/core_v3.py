# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import itertools
import os
import logging
import inspect

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
from sting.modules.simulation_emt.utils import modify_user_functions

import numpy as np
from numba import njit
# class_id: array of ints mapping class to int
# component_id: array of ints mapping instance of class to int

from enum import IntEnum
from numba import njit

class ClassID(IntEnum):
    PARALLEL_RC_SHUNT_2A = 0
    SERIES_RL_BRANCH_2A = 1
    IMPEDANCE_LOAD = 2
    VOLTAGE_SOURCE_4A = 3
    # ...

from sting.datasets.toy_2 import toy_2
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
#numba.njit(f)
from sting.utils.dynamical_systems import TimeDomainSolution

# Set up logging
logger = logging.getLogger(__name__)

# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class SimulationEMT:
    # Dynamic variables
    inputs: DynamicalVariables
    states: DynamicalVariables
    outputs: DynamicalVariables

    components: list = None

    # Interconnections
    ccm_matrices: list[np.ndarray] = None

    # Dict
    x_idx: dict[str, np.ndarray] = None
    u_idx: dict[str, np.ndarray] = None
    

    def __post_init__(self):
        self.build_dictionaries()
        

    @classmethod
    def from_system(cls, system) -> 'SimulationEMT':

        # 0. Make sure EMT attributes have been defined for system components
        system.apply("_calculate_emt_initial_conditions")
        system.apply("define_variables_emt")
        
        # 1. Select all EMT components
        components = system.query(["ccm_generators", "ccm_shunts", "ccm_branches"]).to_list()

        # 2. Get all component data + class_ids + component_ids
        # TODO:

        # 3. Set up inputs, outputs, and states
        states = sum([c.variables_emt.x for c in components], DynamicalVariables(name=[]))
        outputs = sum([c.variables_emt.y for c in components], DynamicalVariables(name=[]))
        inputs = sum([c.variables_emt.u for c in components], DynamicalVariables(name=[]))
        inputs_device = inputs[inputs.type == "device"]
        inputs_grid = inputs[inputs.type == "grid"]
        inputs = inputs_device + inputs_grid

        # 4. Build CCM matrices
        ccm_matrices = get_ccm_matrices(system, attribute="variables_emt", dimI=3)

        return SimulationEMT(
            inputs=inputs, 
            states=states, 
            outputs=outputs, 
            ccm_matrices=ccm_matrices,
            components=components)

    @timeit
    def make_system_step(self, get_device_inputs):

        def step(t, x):
            system_step(t, x, get_device_inputs, self.class_ids, ...)

        # Call step once to precompile
        step(0, self.states.init)

        return step
        

    @timeit
    def simulate(self, t_max, input_signals, settings=None):
        """
        Run the EMT simulation for the system.
        """
        if settings is None:
            settings = {'dense_output': True, 'method': 'Radau', 'max_step': 0.001}

        system_step = self.make_system_step(input_signals)

        solution = solve_ivp(
            system_step, 
            [0, t_max], 
            self.states.init, 
            dense_output=settings['dense_output'], 
            method=settings['method'], 
            max_step=settings['max_step'])

        # Define timepoints that will be used to evaluate the solution of the ODEs
        if settings['dense_output']:
            tps = np.linspace(0, t_max, 500)
            solution = solution.sol(tps)

        # Set the value of the EMT variables based on the solution of the ODEs
        self.set_value(tps, solution, "x")


    def build_dictionaries(self):
        """
        Define EMT variables for all components in the system
        """
        
        # Create a dictionary to map component names to their corresponding indices in the x, u, and y variables
        # For example, {'voltage_source_4a_0': [0, 1, 2, 3], 'gfmi_18a_0': [4, 5, 6, 7, 8]}
        self.x_idx = {}
        self.u_idx = {}
        #self.ud_idx = {}
        #self.y_idx = {}

        for i, component_name in enumerate(self.states.component):
            self.x_idx.setdefault(component_name, []).append(i)

        for i, component_name in enumerate(self.inputs.component):
            self.u_idx.setdefault(component_name, []).append(i)

        # for i, component_name in enumerate(ud.component):
        #    self.ud_idx.setdefault(component_name, []).append(i)

        #for i, component_name in enumerate(self.outputs.component):
        #    self.y_idx.setdefault(component_name, []).append(i)

        # Create a dictionary: {'voltage_source_4a_0': {i_bus_a : [1]}, 'gfmi_18a_0': {i_bus_c : [2]}}
        # so we can use xs_idx['voltage_source_4a_0']['i_bus_a']
        self.xs_idx = {}
        for i, xs in enumerate(self.states):
            component_name = xs.component[0]
            state_name = xs.name[0]
            self.xs_idx.setdefault(component_name, {})[state_name] = i

        # Create a dictionary: {'voltage_source_4a_0': {v_ref_d : [1]}, 'gfmi_18a_0': {p_ref : [2]}}
        self.us_idx ={}
        for i, us in enumerate(self.inputs):
            component_name = us.component[0]
            input_name = us.name[0]
            self.us_idx.setdefault(component_name, {})[input_name] = i

    def plot_results(self, components = None, output_directory =None):
        """
        Plot EMT simulation results
        """

        if components is None:
            components = self.components

        logger.info(f" - Plotting EMT simulation results in {output_directory}")

        for c in components:
            results = c.plot_results_emt()
            results.to_plotly(figure_filepath=os.path.join(output_directory, f"{c.type_}_{c.id}.html"))
    
    def write_results_csv(self, components = None, output_directory=None):
        """
        Write EMT simulation results to output directory.
        """

        if components is None:
            components = self.components

        logger.info(f" - Writing EMT simulation results in {output_directory}")

        for c in components:
            results = c.plot_results_emt()
            results.to_timeseries(csv_filepath=os.path.join(output_directory, f"{c.type_}_{c.id}.csv"))

    def set_value(self, time, numerical_vector, var_type: str):
        """
        Update the value of the EMT variables based on a numerical vector
        """

        for c in self.components:
            variables = c.variables_emt
            x_idx = self.x_idx[c.type_ + "_" + str(c.id)]
            value = numerical_vector[x_idx]

            var_component = getattr(variables, var_type)
            setattr(var_component, "value", value)
            setattr(var_component, "time", time)


# --------------------------------------
# Numba compiled functions
# --------------------------------------

@njit
def derivative_dispatcher(x, u, data, class_id):
    # Start with most common elements to increase hits
    if class_id == ClassID.PARALLEL_RC_SHUNT_2A:
        pass

    elif class_id == ClassID.SERIES_RL_BRANCH_2A:
        pass

    elif class_id == ClassID.IMPEDANCE_LOAD:
        pass

    elif class_id == ClassID.VOLTAGE_SOURCE_4A:
        return voltage_source_4a_dxdt(x, u, data)

@njit
def output_dispatcher(x, class_id):
    pass


@njit
def system_step(
    t, x, 
    get_device_inputs,
    class_ids,
    component_ids,
    component_data, # Lumpy...
    x_ids,
    u_ids, # Lumpy...
    F,
    G):

    # Build device input
    u_device = get_device_inputs(x, t)

    # Build output
    y_stack = np.empty(F.shape[1], dtype=np.float64)

    for i in range(len(component_ids)):
        start, stop = x_ids[i]
        class_id = class_ids[i]
        y_stack[start:stop] = output_dispatcher(x[start:stop], class_id)

    u = F @ y_stack + G @ u_device

    # Build state derivative
    dx_dt = np.empty_like(x)

    for i in range(len(component_ids)):
        x_start, x_stop = x_ids[i]
        x_i = x[x_start:x_stop]
        u_i = u[u_ids[i]]
        class_id = class_ids[i]
        data_i = component_data[class_id][component_ids[i]]
        
        dx_dt[x_start:x_stop] = derivative_dispatcher(x_i, u_i, data_i, class_id)
    
    return dx_dt