# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
import itertools
from more_itertools import transpose
from typing import NamedTuple, Optional, ClassVar
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
import sting.system.selections as sl
from sting.utils.dynamical_systems import DynamicalVariables
from sting.utils.graph_matrices import get_ccm_matrices

# -----------
# Sub-classes
# -----------
class VariablesEMT(NamedTuple):
    """
    All variables used for simulation. 
    """
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables

class ComponentEMT(NamedTuple):
    """
    A component of the system that participates in simulation.

    #### Attributes:
    - type: `str`
            inf_src, se_rl, pa_rc, ... etc. 
    - idx: `int`
            Index of the component in its corresponding list in the system.
    """
    type: str
    idx: int

# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class SimulationEMT:
    """
    Class to simulate the EMT dynamics of a power system.

    #### Attributes:
    - system: `System`
            The system to be simulated.
    - case_directory: `str`
            Path to the case study directory.
    - components: `list[ComponentEMT]`
            List of components that participate in the EMT simulation.
    - variables: `VariablesEMT`
            All variables used for simulation.
    - ccm_abc_matrices: `list[np.ndarray]`
            List of CCM matrices in abc frame.
    """
    system: System
    case_directory: str
    components: list[ComponentEMT] = field(init=False)
    variables: VariablesEMT = field(init=False)
    ccm_abc_matrices: list[np.ndarray] = field(init=False)

    def __post_init__(self):
        self.get_components()
        self.get_variables()
        self.assign_idx()
        self.get_ccm_matrices()

    def get_components(self):
        """
        Get components that qualified for building the differential equations for EMT dynamics.
        Not all components in system, e.g., bus, line_pi, etc., participate in EMT simulation.         
        """

        components = []
        for component in self.system:
            if (    hasattr(component, "idx_variables_emt") 
                and hasattr(component, "define_variables_emt")
                and hasattr(component, "get_derivative_state_emt")
                and hasattr(component, "get_output_emt")
                and hasattr(component, "plot_results_emt")
                ):
                components.append(ComponentEMT(type = component.type, idx = component.idx))
        
        self.components = components
    

    def apply(self, method: str, *args):
        """
        Apply a method to the components for EMT simulation.
        """
        for c in self.components:
               component = getattr(self.system, c.type)[c.idx-1]
               getattr(component, method)(*args)

    def get_variables(self):
        """
        Define EMT variables for all components in the system
        """
        self.apply("define_variables_emt")

        generators, = self.system.generators.select("variables_emt")
        shunts, = self.system.shunts.select("variables_emt")
        branches, = self.system.branches.select("variables_emt")

        variables_emt = itertools.chain(generators, shunts, branches)

        fields = ["x", "u", "y"]
        selection = [[getattr(c, f) for f in fields] for c in variables_emt]
        stack = dict(zip(fields, transpose(selection)))

        x = sum(stack["x"], DynamicalVariables(name=[]))
        y = sum(stack["y"], DynamicalVariables(name=[]))

        u = sum(stack["u"], DynamicalVariables(name=[]))
        ud = u[u.type == "device"]
        ug = u[u.type == "grid"]
        u = ud + ug

        self.variables = VariablesEMT(x=x, u=u, y=y)
    
    def assign_idx(self):
        """
        Assign index, e.g., [False, True, False, ...], of the system-wide variables to each component.
        For example, if the system state vector is [i_bus_a, i_bus_b, i_bus_c, v_bus_a, ...],
        and a component is an infinite source connected to bus 1, then the index for that component
        will be [True, True, True, False, ...], indicating that the first three variables in the system state
        vector correspond to that component.
        """

        x, u, y = self.variables
        for c in self.components:
                component = getattr(self.system, c.type)[c.idx-1]
                id = f"{c.type}_{c.idx}"
                setattr(component, "idx_variables_emt", {   "x": x.component == id, 
                                                            "u": u.component == id,
                                                            "y": y.component == id  })
                                       
    def get_ccm_matrices(self):
        """
        Get the CCM matrices in abc frame for the EMT simulation.
        """
        
        self.ccm_abc_matrices = get_ccm_matrices(self.system, "variables_emt", 3)
    

    def get_input_vector(self, u_signals, t):

        d_vars = self.variables.u[self.variables.u.type == "device"]

        ud = [u_signals[component][name](t) if u_signals.get(component, {}).get(name) else 0 for (component, name) in zip(d_vars.component, d_vars.name)]
        ud = np.array(ud) + d_vars.init

        g_vars = self.variables.u[self.variables.u.type == "grid"]
        ug = np.full(len(g_vars), np.nan, dtype=float)

        u = np.hstack((ud, ug))

        return u, ud
    
    def set_value(self, time, numerical_vector, var_type: str):
        """
        Update the value of the EMT variables based on a numerical vector
        """

        for c in self.components:
            component = getattr(self.system, c.type)[c.idx-1]
            variables = getattr(component, "variables_emt")
            idx = getattr(component, "idx_variables_emt")
            value = numerical_vector[idx[var_type]]

            var_component = getattr(variables, var_type)
            setattr(var_component, "value", value)
            setattr(var_component, "time", time)

    def sim(self, t_max, inputs, settings={'dense_output': True, 'method': 'Radau', 'max_step': 0.001}):
        """
        Simulate the EMT dynamics of the system using scipy.integrate.solve_ivp
        """
        
        F, G, H, L = self.ccm_abc_matrices
        x_len = len(self.variables.x)
        y_len = len(self.variables.y)

        def system_ode(t, x, u_signals):

            u, ud = self.get_input_vector(u_signals, t)

            y_stack = np.full(y_len, np.nan, dtype=float)

            self.set_value(t, x, "x")

            for c in self.components:
                component = getattr(self.system, c.type)[c.idx-1]
                variables = getattr(component, "variables_emt")
                idx = getattr(component, "idx_variables_emt")
                
                # Update input values
                u_component = getattr(variables, "u")
                setattr(u_component, "value", u[idx["u"]])

                # Get output values
                y = getattr(component, "get_output_emt")()
                y_stack[idx["y"]] = y

            ustack = F @ y_stack + G @ ud

            dx_stack =  np.full(x_len, np.nan, dtype=float)

            self.set_value(t, ustack, "u")

            for c in self.components:
                component = getattr(self.system, c.type)[c.idx-1]
                variables = getattr(component, "variables_emt")
                idx = getattr(component, "idx_variables_emt")

                # Get derivative of state
                dx = getattr(component, "get_derivative_state_emt")()
                dx_stack[idx["x"]] = dx

            return dx_stack
        
        solution = solve_ivp(system_ode, 
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

        self.set_value(tps, solution, "x")

        print("EMT Simulation completed.")


    def plot_results(self, components = None):
        """
        Plot EMT simulation results
        """

        if components is None:
            components = self.components

        output_dir = os.path.join(self.case_directory, "outputs", "emt")
        for c in components:
            component = getattr(self.system, c.type)[c.idx-1]
            getattr(component, "plot_results_emt")(output_dir)
    
        
