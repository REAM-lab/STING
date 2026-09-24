from sting.datasets.toy_2 import toy_2
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
#numba.njit(f)

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
    # Run with numba jit?
    jit: bool = False

    # Dynamic variables
    inputs: DynamicalVariables
    states: DynamicalVariables
    outputs: DynamicalVariables

    # Simulation functions
    derivative_steps = None
    output_steps = None

    # Interconnections
    ccm_matricies: list[np.ndarray] = None

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

    def from_system(cls, system, jit=False):

        # 0. Make sure EMT attributes have been defined for system components
        system.apply("_calculate_emt_initial_conditions")
        system.apply("define_variables_emt")
        
        # 1. Select all EMT components
        components = system.query(["ccm_generators", "ccm_shunts", "ccm_branches"]).to_list()

        # 2. For each component build it's step function and output function
        derivative_steps = None
        output_steps = None

        # 3. Set up inputs, outputs, and states
        states = sum([c.variables_emt.x for c in components], DynamicalVariables(name=[]))
        outputs = sum([c.variables_emt.y for c in components], DynamicalVariables(name=[]))
        inputs = sum([c.variables_emt.u for c in components], DynamicalVariables(name=[]))
        inputs_device = inputs[inputs.type == "device"]
        inputs_grid = inputs[inputs.type == "grid"]
        inputs = inputs_device + inputs_grid

        # 4. Build CCM matricies
        ccm_matrices = get_ccm_matrices(system, attribute="variables_emt", dimI=3)

        return

    def make_device_inputs(self, signals):

        def get_device_inputs(x, t, signals=signals, u0=self.inputs.init, us_idx=self.us_idx, xs_idx=self.xs_idx):
            u_device = u0.copy()

            for component in signals: # e.g., component = 'gfmi_18a_0'
                for input, func in signals[component].items(): # e.g., input = 'v_ref_d'
                    ud_idx = us_idx[component][input] # index associated to component and input signal
                    u_device[ud_idx] += func(t, x = x, id = xs_idx) # evaluate function at "t"
            
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
            derivative_steps=self.derivative_steps, 
            output_steps=self.output_steps, 
            x_idx=self.x_idx,
            u_idx=self.u_idx,
            F=self.ccm_matrices[0], 
            G=self.ccm_matrices[1]):
            """
            System step for the EMT simulation.
            """

            # Build device input
            u_device = get_device_inputs(x, t)

            # Build output
            y_stack = sum([g(x[i]) for g, i in zip(output_steps, x_idx)], [])

            u = F @ y_stack + G @ u_device

            # Build state derivative
            dx_dt = sum([f(x[i], u[j]) for f, i, j in zip(derivative_steps, x_idx, u_idx)], [])
                        

            return dx_dt

        if self.jit:
            import numba
            system_step = numba.njit(system_step)
            # Call once to precompile
            system_step(0, self.states.x.init,)

        solution = solve_ivp(
            system_step, 
            [0, t_max], 
            self.states.x.init, 
            dense_output=settings['dense_output'], 
            method=settings['method'], 
            max_step=settings['max_step'])

        return solution



    def build_dictionaries(self):
        """
        Define EMT variables for all components in the system
        """
        
        # Create a dictionary to map component names to their corresponding indices in the x, u, and y variables
        # For example, {'voltage_source_4a_0': [0, 1, 2, 3], 'gfmi_18a_0': [4, 5, 6, 7, 8]}
        self.x_idx = {}
        self.u_idx = {}
        #self.ud_idx = {}
        self.y_idx = {}

        for i, component_name in enumerate(self.states.component):
            self.x_idx.setdefault(component_name, []).append(i)

        for i, component_name in enumerate(self.inputs.component):
            self.u_idx.setdefault(component_name, []).append(i)

        # for i, component_name in enumerate(ud.component):
        #    self.ud_idx.setdefault(component_name, []).append(i)

        for i, component_name in enumerate(self.outputs.component):
            self.y_idx.setdefault(component_name, []).append(i)

        # Create a dictionary: {'voltage_source_4a_0': {i_bus_a : [1]}, 'gfmi_18a_0': {i_bus_c : [2]}}
        # so we can use xs_idx['voltage_source_4a_0']['i_bus_a']
        self.xs_idx ={}
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