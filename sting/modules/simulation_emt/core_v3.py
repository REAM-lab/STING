import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import os
import logging
import numba

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

    # Interconnections
    components: list = None
    ccm_matrices: list[np.ndarray] = None
    
    # Component data
    parallel_rc_shunt_2a: np.ndarray = None
    series_rl_branch_2a: np.ndarray = None
    voltage_source_4a: np.ndarray = None
    gfli_16a: np.ndarray = None
    gfmi_18a: np.ndarray = None

    # EMT simulation step
    step: None = None

    # Dicts
    x_idx: dict = None
    u_idx: dict = None
    
    
    @timeit
    def __post_init__(self):
        """Precompiling EMT simulation"""
        self.build_dictionaries()

        x_index = np.array([min(a) for a in self.x_idx.values()])
        u_len = np.array([len(a) for a in self.u_idx.values()])
        u_stop = np.cumsum(u_len)
        u_start = np.insert(u_stop[:-1], 0, 0)
        u_index = np.array(list(zip(u_start, u_stop)))
        u_values = np.array(sum([a for a in self.u_idx.values()], []))

        u0 =  self.inputs[self.inputs.type == "device"].init

        self.step = lambda t, x, inputs: system_step(
            t, x, 
            inputs,
            # Vector indices
            x_index=x_index,
            u_index=u_index,
            u_values=u_values,
            u0=u0,
            # Interconnections
            F=self.ccm_matrices[0],
            G=self.ccm_matrices[1],
            # Component data
            parallel_rc_shunt_2a=self.parallel_rc_shunt_2a,
            series_rl_branch_2a=self.series_rl_branch_2a,
            voltage_source_4a=self.voltage_source_4a,
            gfli_16a=self.gfli_16a,
            gfmi_18a=self.gfmi_18a
            )

        # Step once to precompile 
        self.step(t=0, x=self.states.init, inputs=numba.njit(lambda t, x: u0))
        

    @classmethod
    def from_system(cls, system) -> 'SimulationEMT':

        # 0. Make sure EMT attributes have been defined for system components
        system.apply("_calculate_emt_initial_conditions")
        system.apply("define_variables_emt")
        
        # 1. Select all EMT components
        components = system.query(["ccm_generators", "ccm_shunts", "ccm_branches"]).to_list()

        # 2. Set up inputs, outputs, and states
        states = sum([c.variables_emt.x for c in components], DynamicalVariables(name=[]))
        outputs = sum([c.variables_emt.y for c in components], DynamicalVariables(name=[]))
        inputs = sum([c.variables_emt.u for c in components], DynamicalVariables(name=[]))
        inputs_device = inputs[inputs.type == "device"]
        inputs_grid = inputs[inputs.type == "grid"]
        inputs = inputs_device + inputs_grid

        # 3. Build CCM matrices
        ccm_matrices = get_ccm_matrices(system, attribute="variables_emt", dimI=3)

        # 4. Get component data
        parallel_rc_shunt_2a = system.query(["parallel_rc_shunt_2a"]).to_table("g_pu", "b_pu", "wbase").to_numpy()
        series_rl_branch_2a = system.query(["series_rl_branch_2a"]).to_table("r_pu", "x_pu", "wbase").to_numpy()
        voltage_source_4a = system.query(["voltage_source_4a"]).to_table("r_pu", "x_pu", "wbase").to_numpy()
        gfli_16a = system.query(["gfli_16a"]).to_table(
            "rf1_pu", "xf1_pu", "rf2_pu", "xf2_pu", "rsh_pu", "csh_pu", 
            "kp_pll_rad_s", "ki_pll_rad2_s2", "tau_pll_s",
            "kp_cc_pu", "ki_cc_puHz", "kff_cc",
            "kp_pc_pu", "ki_pc_puHz",
            "wbase"
        ).to_numpy()

        gfmi_18a = system.query(["gfmi_18a"]).to_table(
            "rf1_pu", "xf1_pu", "rf2_pu", "xf2_pu", "rsh_pu", "csh_pu", 
            "k_q_pu", "w_q_puHz",
            "kp_vc_pu", "ki_vc_puHz", "kffi_vc",
            "kp_cc_pu", "ki_cc_puHz", "kffv_cc",
            "kd_pu", "h_s",
            "wbase"
        ).to_numpy()

        return SimulationEMT(
            inputs=inputs, 
            states=states, 
            outputs=outputs, 
            ccm_matrices=ccm_matrices,
            components=components,
            parallel_rc_shunt_2a=parallel_rc_shunt_2a,
            series_rl_branch_2a=series_rl_branch_2a,
            voltage_source_4a=voltage_source_4a,
            gfli_16a=gfli_16a,
            gfmi_18a=gfmi_18a
            )
        

    @timeit
    def simulate(self, t_max, inputs, settings=None):
        """
        Run the EMT simulation for the system.
        """
        if settings is None:
            settings = {'dense_output': True, 'method': 'Radau', 'max_step': 0.001}

        solution = solve_ivp(
            self.step, 
            [0, t_max], 
            self.states.init, 
            args=(inputs,),
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

        for i, component_name in enumerate(self.states.component):
            self.x_idx.setdefault(component_name, []).append(i)

        for i, component_name in enumerate(self.inputs.component):
            self.u_idx.setdefault(component_name, []).append(i)


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

from sting.branch.series_rl_branch_2a import series_rl_branch_2a_dxdt
from sting.shunt.parallel_rc_shunt_2a import parallel_rc_shunt_2a_dxdt
from sting.generator.voltage_source_4a import voltage_source_4a_dxdt
from sting.generator.gfli_16a import gfli_16a_dxdt
from sting.generator.gfmi_18a import gfmi_18a_dxdt

#@njit
def derivative_dispatcher(x, x_index, u, u_index, u_values, parallel_rc_shunt_2a, series_rl_branch_2a, voltage_source_4a, gfli_16a, gfmi_18a):
    dx_dt = np.empty_like(x)
    i = 0

    for j in range(voltage_source_4a.shape[0]):
        offset = x_index[i]
        u_i = get_component_u(i, u, u_index, u_values)

        voltage_source_4a_dxdt(x, u_i, dx_dt, voltage_source_4a[j], offset)
        i += 1

    # Generators
    for j in range(gfli_16a.shape[0]):
        offset = x_index[i]
        u_i = get_component_u(i, u, u_index, u_values)

        gfli_16a_dxdt(x, u_i, dx_dt, gfli_16a[j], offset)
        i += 1

    for j in range(gfmi_18a.shape[0]):
        offset = x_index[i]
        u_i = get_component_u(i, u, u_index, u_values)

        gfmi_18a_dxdt(x, u_i, dx_dt, gfmi_18a[j], offset)
        i += 1

    # Shunts
    for j in range(parallel_rc_shunt_2a.shape[0]):
        offset = x_index[i]
        u_i = get_component_u(i, u, u_index, u_values)

        parallel_rc_shunt_2a_dxdt(x, u_i, dx_dt, parallel_rc_shunt_2a[j], offset)
        i += 1

    # Branches
    for j in range(series_rl_branch_2a.shape[0]):
        offset = x_index[i]
        u_i = get_component_u(i, u, u_index, u_values)

        series_rl_branch_2a_dxdt(x, u_i, dx_dt, series_rl_branch_2a[j], offset)
        i += 1

    return dx_dt

#@njit
def get_component_u(i, u, u_index, u_values):
    u_start, u_stop = u_index[i]
    u_i = u[u_values[u_start:u_stop]]

    return u_i

#@njit
def output_dispatcher(x, x_index, n_outputs, parallel_rc_shunt_2a, series_rl_branch_2a, voltage_source_4a, gfli_16a, gfmi_18a):
    
    y_stack = np.empty(n_outputs, dtype=x.dtype)
    i = 0
    offset = 0

    for _ in range(voltage_source_4a.shape[0]):
        start = x_index[i]
        y_stack[offset:offset+3] = x[start:start+3]
        i += 1
        offset += 3

    # Generator outputs
    for _ in range(gfli_16a.shape[0]):
        start = x_index[i]
        # Take the last three states
        y_stack[offset:offset+3] = x[start+13:start+16]
        i += 1
        offset += 3

    for _ in range(gfmi_18a.shape[0]):
        start = x_index[i]
        # Take the last three states
        y_stack[offset:offset+3] = x[start+13:start+16]
        i += 1
        offset += 3

    # Shunt outputs
    for _ in range(parallel_rc_shunt_2a.shape[0]):
        start = x_index[i]
        y_stack[offset:offset+3] = x[start:start+3]
        i += 1
        offset += 3

    # Branch outputs
    for _ in range(series_rl_branch_2a.shape[0]):
        start = x_index[i]
        y_stack[offset:offset+3] = x[start:start+3]
        i += 1
        offset += 3

    return y_stack


#@njit
def system_step(
    t, x, 
    inputs,
    # Vector indices
    x_index,
    u_index,
    u_values,
    u0,
    # Interconnections
    F,
    G,
    # Component data
    parallel_rc_shunt_2a,
    series_rl_branch_2a,
    voltage_source_4a,
    gfli_16a,
    gfmi_18a,
    ):

    # Build device input
    u_device = inputs(t, x) + u0

    # Build output
    n_outputs = F.shape[1]
    y_stack = output_dispatcher(x, x_index, n_outputs, parallel_rc_shunt_2a, series_rl_branch_2a, voltage_source_4a, gfli_16a, gfmi_18a)

    u = F @ y_stack + G @ u_device

    # Build state derivative
    dx_dt = derivative_dispatcher(x, x_index, u, u_index, u_values, parallel_rc_shunt_2a, series_rl_branch_2a, voltage_source_4a, gfli_16a, gfmi_18a)
       
    return dx_dt