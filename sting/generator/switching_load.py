"""
This module implements a passive RL load that switches on and off at specified times. 
- Series RL branch: It is a branch connected to ground.
"""
# -------------
# Import python packages
# --------------
import numpy as np
from dataclasses import dataclass

# -------------
# Import sting code
# -------------
from sting.utils.dynamical_systems import DynamicalVariables
from sting.utils.transformations import abc2dq0
from sting.generator.core import Generator
from sting.modules.simulation_emt.utils import VariablesEMT

# -------------
# Main class
# -------------
@dataclass(slots=True, kw_only=True, eq=False)
class SwitchingLoad(Generator):
    r_pu: float
    x_pu: float
    ton_sec: float
    toff_sec: float

    def define_variables_emt(self):
        
        # States
        # ------

        # Initial conditions
        x = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[0, 0, 0],
        )

        # Inputs
        # ------

        u = DynamicalVariables(
            name=["v_ground", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "grid", "grid", "grid"],
            init=[0, 0, 0, 0],
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
        )

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)
    
    def get_derivative_state_emt(self):

        # Get state values
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value

        # Get input values
        t = self.variables_emt.x.time
        if self.ton_sec <= t < self.toff_sec:
            v_ground, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value
        else:
            v_ground, v_bus_a, v_bus_b, v_bus_c = 0, 0, 0, 0

        # Get parameters
        r = self.r_pu
        x = self.x_pu
        wb = 2 * np.pi * self.base_frequency_Hz

        # Differential equations
        d_i_bus_a = wb / x * (v_ground - v_bus_a - r * i_bus_a)
        d_i_bus_b = wb / x * (v_ground - v_bus_b - r * i_bus_b)
        d_i_bus_c = wb / x * (v_ground - v_bus_c - r * i_bus_c)

        return [d_i_bus_a, d_i_bus_b, d_i_bus_c]
    
    def get_output_emt(self):
        
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value

        return [i_bus_a, i_bus_b, i_bus_c]
    
    def plot_results_emt(self):
        """
        Plot EMT simulation results
        """

        # Get time
        time = self.variables_emt.x.time
        angle_ref =  2 * np.pi * self.base_frequency_Hz * time

        # Retrieve state values
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value
        
        i_bus_d, i_bus_q, _ = zip(*map(abc2dq0, i_bus_a, i_bus_b, i_bus_c, angle_ref))

        results = DynamicalVariables(
            name=["i_bus_d", "i_bus_q"],
            component=f"{self.type_}_{self.id}",
            value=[i_bus_d, i_bus_q],
            time=time,
        )
        return results