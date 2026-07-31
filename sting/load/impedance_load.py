"""
This module implements a passive RL load.
- Series RL branch: It is a branch connected to ground.
"""
# -------------
# Import python packages
# --------------
import numpy as np
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

# -------------
# Import sting code
# -------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.utils.transformations import abc2dq0
from sting.load.core import Load
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.transformations import dq02abc, abc2dq0
from sting.modules.power_flow.utils import ACPowerFlowSolution

# -------------
# Sub-classes
# -------------
class InitialConditionsEMT(NamedTuple):
    v_bus_D: float
    v_bus_Q: float
    i_bus_D: float
    i_bus_Q: float

# ----------------
# Sub-classes
# ----------------
class PowerFlowVariables(NamedTuple):
    p_bus: float
    q_bus: float
    vmag_bus: float
    vphase_bus: float

# -------------
# Main class
# -------------
@dataclass(slots=True, kw_only=True, eq=False)
class ConstantImpedanceLoad(Load):
    r_pu: float = None
    x_pu: float = None
    power_flow_variables: PowerFlowVariables = None
    emt_init: InitialConditionsEMT = None
    tags: ClassVar[list[str]] = ["ccm_generator"] # When building the CCM interconnection matrices, we need to consider the load as a generator.
    

    def load_ac_power_flow_solution(self, timepoint: str, pf_solution: ACPowerFlowSolution):

        if self.timepoint != timepoint:
            return  # Skip if the timepoint does not match

        self.power_flow_variables = PowerFlowVariables(
            p_bus=-self.load_MW/self.base_power_MVA,
            q_bus=-self.load_MVAR/self.base_power_MVA,
            vmag_bus=pf_solution.bus_voltage_magnitude[self.bus_id, timepoint],
            vphase_bus=pf_solution.bus_voltage_angle[self.bus_id, timepoint],
        )

        # Compute parameters r_pu and x_pu based on the power flow solution
        vmag = self.power_flow_variables.vmag_bus
        angle = self.power_flow_variables.vphase_bus

        v = vmag * np.exp(1j * angle * np.pi / 180)
        s = -self.power_flow_variables.p_bus - 1j * self.power_flow_variables.q_bus
        i = np.conj(s / v)
        z = v / i

        self.r_pu = z.real
        self.x_pu = z.imag

    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.power_flow_variables.vmag_bus
        vphase_bus = self.power_flow_variables.vphase_bus

        v_bus_DQ = vmag_bus * np.exp(vphase_bus * 1j * np.pi / 180)
        i_bus_DQ = -v_bus_DQ / (self.r_pu + 1j * self.x_pu) # current flowing out of the load (into the grid)

        self.emt_init = InitialConditionsEMT(
            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag,
            i_bus_D=i_bus_DQ.real,
            i_bus_Q=i_bus_DQ.imag
        )

    def _build_small_signal_model(self):

        r = self.r_pu
        x = self.x_pu

        wb = 2 * np.pi * self.base_frequency_Hz

        # Define state-space matrices (turn off code formatters for matrices)
        # fmt: off
        A = wb * np.array(
            [[-r/x,    1], 
             [  -1, -r/x]])

        B = wb * np.array(
            [[1/x, 0,    -1/x,      0], 
             [    0, 1/x, 0, -1/x]])
        # fmt: on
        C = np.eye(2)

        D = np.zeros((2, 4))

        u = DynamicalVariables(
            name=["v_ground_D", "v_ground_Q", "v_bus_D", "v_bus_Q"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "grid", "grid"],
            init=[
                0.0,
                0.0,
                self.emt_init.v_bus_D,
                self.emt_init.v_bus_Q,
            ],
        )

        x = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            component=f"{self.type_}_{self.id}",
            init=[self.emt_init.i_bus_D, self.emt_init.i_bus_Q],
        )

        y = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            component=f"{self.type_}_{self.id}",
            init=[self.emt_init.i_bus_D, self.emt_init.i_bus_Q],
        )

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)

    def define_variables_emt(self):
        
        # States
        # ------

        # Initial conditions
        i_bus_D, i_bus_Q = self.emt_init.i_bus_D, self.emt_init.i_bus_Q
        i_bus_a, i_bus_b, i_bus_c = dq02abc(i_bus_D, i_bus_Q, 0, 0)

        x = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type_}_{self.id}",
            init=[i_bus_a, i_bus_b, i_bus_c],
        )

        # Inputs
        # ------
        # Initial conditions
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, 0)

        u = DynamicalVariables(
            name=["v_ground_a", "v_ground_b", "v_ground_c", "v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type_}_{self.id}",
            type=["device", "device", "device", "grid", "grid", "grid"],
            init=[0.0, 0.0, 0.0, v_bus_a, v_bus_b, v_bus_c],
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
        v_ground_a, v_ground_b, v_ground_c, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        # Get parameters
        r = self.r_pu
        x = self.x_pu
        wb = 2 * np.pi * self.base_frequency_Hz

        # Differential equations
        d_i_bus_a = wb / x * (v_ground_a - v_bus_a - r * i_bus_a)
        d_i_bus_b = wb / x * (v_ground_b - v_bus_b - r * i_bus_b)
        d_i_bus_c = wb / x * (v_ground_c - v_bus_c - r * i_bus_c)

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
        angle =  2 * np.pi * self.base_frequency_Hz * time

        # Retrieve state values
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value
        
        i_bus_D, i_bus_Q, _ = zip(*map(abc2dq0, i_bus_a, i_bus_b, i_bus_c, angle))
        p = self.r_pu * (np.array(i_bus_D) ** 2 + np.array(i_bus_Q) ** 2)
        q = self.x_pu * (np.array(i_bus_D) ** 2 + np.array(i_bus_Q) ** 2)

        results = DynamicalVariables(
            name=["p", "q", "i_bus_D", "i_bus_Q"],
            component=f"{self.type_}_{self.id}",
            value=[p, q, i_bus_D, i_bus_Q],
            time=time,
        )
        return results