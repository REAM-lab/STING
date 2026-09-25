from sting.datasets.toy_2 import toy_2
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
#numba.njit(f)
from sting.utils.dynamical_systems import TimeDomainSolution

"""sys = toy_2()
sys.apply("post_system_init", sys)
# Run power flow
pf = ACPowerFlow(system=sys)
pf.solve()

# Break down lines into branches and shunts for small-signal modeling
sys_modifier = SystemModifier(system=sys)
sys_modifier.decompose_lines()
sys_modifier.combine_shunts()
sys_modifier.create_impedance_loads()

print("ok")"""

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

    # Run with numba jit?
    jit: bool = False

    components: list = None

    # Simulation functions
    derivative_steps:list = None
    output_steps:list = None

    # Interconnections
    ccm_matrices: list[np.ndarray] = None

    # Dictonaries
    x_idx: dict[str, np.ndarray] = None
    u_idx: dict[str, np.ndarray] = None
    xs_idx: dict[str, dict[str, int]] = None
    us_idx: dict[str, dict[str, int]] = None
    
    #ud_idx: dict[str, np.ndarray] = None
    

    def __post_init__(self):

        self.build_dictionaries()
        
        if self.jit:
            import numba
            self.derivative_steps = [numba.njit(f) for f in self.derivative_steps]
            self.output_steps = [numba.njit(g) for g in self.output_steps]

    @classmethod
    def from_system(cls, system, jit=False) -> 'SimulationEMT':

        # 0. Make sure EMT attributes have been defined for system components
        system.apply("_calculate_emt_initial_conditions")
        system.apply("define_variables_emt")
        
        # 1. Select all EMT components
        components = system.query(["ccm_generators", "ccm_shunts", "ccm_branches"]).to_list()

        # 2. For each component build it's step function and output function
        derivative_steps = [c.make_derivative_state_emt(jit) for c in components]
        output_steps = [c.make_output_emt(jit) for c in components]

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
            jit=jit, 
            inputs=inputs, 
            states=states, 
            outputs=outputs, 
            ccm_matrices=ccm_matrices, 
            derivative_steps=derivative_steps, 
            output_steps=output_steps,
            components=components)

    def make_device_inputs(self, signals):
        # Device initial conditions
        u0 = self.inputs[self.inputs.type == "device"].init
        # Input signal functions and index
        u_func = []
        u_idx =[]

        for component in signals: # e.g., component = 'gfmi_18a_0'
            for input, func in signals[component].items(): # e.g., input = 'v_ref_d'
                u_id = self.us_idx[component][input] # index associated to component and input signal
                u_func.append(func)
                u_idx.append(u_id)

        def get_device_inputs(x, t, u0=u0, u_idx=tuple(u_idx), u_func=tuple(u_func)):
            u_device = u0.copy()

            for u_id, func in zip(u_idx, u_func):
                u_device[u_id] += func(t, x = x)
            
            return u_device

        if self.jit:
            import numba
            return numba.njit(get_device_inputs)
            
        return get_device_inputs


    @timeit
    def simulate(self, t_max, input_signals, settings=None):
        """
        Run the EMT simulation for the system.
        """

        # TODO: We are assuming dx_dt and y_stack are contig, this may be incorrect.

        if settings is None:
            settings = {'dense_output': True, 'method': 'Radau', 'max_step': 0.001}

        get_device_inputs = self.make_device_inputs(input_signals)

        def system_step(
            t, x, 
            get_device_inputs=get_device_inputs,
            derivative_steps=tuple(self.derivative_steps), 
            output_steps=tuple(self.output_steps), 
            x_idx=tuple([np.array(l) for l in self.x_idx.values()]),
            u_idx=tuple([np.array(l) for l in self.u_idx.values()]),
            F=self.ccm_matrices[0], 
            G=self.ccm_matrices[1]):
            """
            System step for the EMT simulation.
            """

            # Build device input
            u_device = get_device_inputs(x, t)

            # Build output
            y_stack = np.empty(F.shape[1], dtype=x.dtype)

            offset = 0
            for k in range(len(output_steps)):
                i = x_idx[k]
                g = output_steps[k]
                y = g(x[i])
                n = y.shape[0]
                y_stack[offset:offset+n] = y
                offset += n

            u = F @ y_stack + G @ u_device

            # Build state derivative
            dx_dt = np.empty_like(x)

            for f, i, j in zip(derivative_steps, x_idx, u_idx):
                dx_dt[i] = np.array(f(x[i], u[j]))
            
            return dx_dt

        if self.jit:
            import numba
            # system_step = numba.njit(system_step)
            # Call once to precompile
            system_step(0, self.states.init)

        @timeit
        def solve_emt():
            """Solve the ODEs"""

            solution = solve_ivp(
                system_step, 
                [0, t_max], 
                self.states.init, 
                dense_output=settings['dense_output'], 
                method=settings['method'], 
                max_step=settings['max_step'])
            
            return solution

        solution = solve_emt()

        # Define timepoints that will be used to evaluate the solution of the ODEs
        if settings['dense_output']:
            tps = np.linspace(0, t_max, 500)
            solution = solution.sol(tps)

        # Set the value of the EMT variables based on the solution of the ODEs
        self.set_value(tps, solution, "x")


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