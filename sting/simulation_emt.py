# Import python packages
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
import itertools
from more_itertools import transpose
from typing import NamedTuple, Optional, ClassVar

# Import sting code
from sting.system.core import System
import sting.system.selections as sl
from sting.utils.dynamical_systems import DynamicalVariables
from sting.utils.graph_matrices import get_ccm_matrices

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables


@dataclass(slots=True)
class SimulationEMT:
    system: System
    variables_emt: VariablesEMT = field(init=False)
    ccm_abc_matrices: list[np.ndarray] = field(init=False)

    
    def define_variables(self):
        """
        Define EMT variables for all components in the system
        """
        self.system.apply("_define_variables_emt")

        generators, = self.system.generators.select("var_emt")
        shunts, = self.system.shunts.select("var_emt")
        branches, = self.system.branches.select("var_emt")

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

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)
    
    def assign_idx_solveivp(self):

        x, u, y = self.variables_emt
        for component in self.system:
            if hasattr(component, "idx_solve_ivp"):
                id = f"{component.type}_{component.idx}"
                component.idx_solve_ivp = {"x": x.component == id, 
                                           "u": u.component == id,
                                           "y": y.component == id}

        #components = np.unique(x.component).tolist()
        #components = [[c, c.rpartition('_')[0], int(c.rpartition('_')[2])] for c in components]
        #for c  in components:
        #    c.append([x.component == c[0]])
        #    c.append((u.component == c[0]))


        # [['inf_src', '1', [...]], ['inf_src', '2', [...]], ['pa_rc', '1', [...]], ['pa_rc', '2', [...]], ['se_rl', '1', [...]]]

        #self.components = components
        #self.x = x
        #self.u = u
        #self.y = y

        self.ccm_abc_matrices = get_ccm_matrices(self.system, "var_emt", 3)
    
    def update_components_variables(self, vector: np.ndarray, variable: str):

        for component in self.system:
            if hasattr(component, "idx_solve_ivp"):
                v = getattr(getattr(component, "var_emt"), variable)
                setattr(v, "value", vector[component.idx_solve_ivp[variable]])


    def get_stack_vector(self, variable: str, method: str, *args):

        length =  len(getattr(self.variables_emt, variable))
        vector = np.full(length, np.nan, dtype=float)

        for component in self.system:
            if hasattr(component, "idx_solve_ivp"):
                idx = component.idx_solve_ivp[variable]
                v = getattr(component, method, *args)
                vector[idx] = v

        return vector
        #for _, component_type, idx, x_idx, u_idx in self.components:
            # Get component
        #    component = getattr(self.system, component_type)[idx-1]
        #    variables = getattr(component, "var_emt")

            # Update state values
        #    x_component = getattr(variables, "x")
        #    setattr(x_component, "value", x[x_idx])






    def sim(self, t_max, inputs):
        """
        Simulate the EMT dynamics of the system using scipy.integrate.solve_ivp
        """
        
        F, G, H, L = self.ccm_abc_matrices

        
        def system_ode(t, x, u_signals):

            angle_sys = x[-1]  # last state is system angle

            ud = self.u[self.u.type == "device"]
            ud_vals = [u_signals[component][name](t) if u_signals.get(component, {}).get(name) else 0 for (component, name) in zip(ud.component, ud.name)]
            ud_vals = np.array(ud_vals) + ud.init

            ug = self.u[self.u.type == "grid"]
            ug_vals = np.zeros_like(len(ug))

            u = np.vstack((ud_vals, ug_vals))

            y_stack = []

            self.update_components_variables(x, "x")
            self.update_components_variables(u, "u")
            y_stack = self.get_stack_vector("y", "_get_output_emt", t)

            ustack = F @ y_stack + G @ ud_vals

            self.update_components_variables(ustack, "u")
            dx = self.get_stack_vector("x", "_get_derivative_state_emt", t, angle_sys)

            d_angle_sys = 2 * np.pi * 60 # rad/s
            dx = np.append(dx, [d_angle_sys])

            return dx

            #for _, component_type, idx, x_idx, u_idx in self.components:
                # Get component
            #    component = getattr(self.system, component_type)[idx-1]
            #    variables = getattr(component, "var_emt")
                
                # Update state values
            #    x_component = getattr(variables, "x")
            #    setattr(x_component, "value", x[x_idx])

                # Update input values
            #    u_component = getattr(variables, "u")
            #    setattr(u_component, "value", u[u_idx])

                # Get output values
            #    y = getattr(component, "_get_output_emt")(t)
            #    y_stack.extend(y)

            #y_stack = np.array(y_stack).flatten()

            ustack = F @ y_stack + G @ ud_vals

            dx = []
        
            for _, component_type, idx, x_idx, ug_idx in self.components:
                # Get component
                component = getattr(self.system, component_type)[idx-1]
                variables = getattr(component, "var_emt")

                # Update input values
                u_component = getattr(variables, "u")
                setattr(u_component, "value", ustack[u_idx])

                # Get derivative of state
                dx_comp = getattr(component, "_get_derivative_state_emt")(t, angle_sys)
                dx.extend(dx_comp)

            d_angle_sys = 2 * np.pi * 60 # rad/s
            dx.append(d_angle_sys)

            dx = np.array(dx).flatten()

            return dx
        
        x_init = self.x.init
        x_init = np.append(x_init, [0.0])  # initial system angle

        solution = solve_ivp(system_ode, 
                        [0, t_max], # timeperiod 
                        x_init, # initial conditions
                        dense_output=True,  
                        args=(inputs, ),
                        method='Radau', 
                        max_step=0.001)
        
        return solution