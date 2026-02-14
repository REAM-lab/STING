# Import python packages
from dataclasses import dataclass, field
from sting.utils.transformations import dq02abc, abc2dq0
from typing import NamedTuple, Optional, ClassVar
import numpy as np
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import sting code
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.modules.power_flow.utils import ACPowerFlowSolution


class PowerFlowVariables(NamedTuple):
    vmag_from_bus: float
    vphase_from_bus: float
    vmag_to_bus: float
    vphase_to_bus: float


class InitialConditionsEMT(NamedTuple):
    vmag_from_bus: float
    vphase_from_bus: float
    vmag_to_bus: float
    vphase_to_bus: float
    v_from_bus_D: float
    v_from_bus_Q: float
    v_to_bus_D: float
    v_to_bus_Q: float
    i_br_D: float
    i_br_Q: float

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables

@dataclass(slots=True)
class BranchSeriesRL:
    id: int = field(default=-1, init=False)
    name: str 
    from_bus: str
    to_bus: str
    base_power_MVA: float
    base_voltage_kV: float
    base_frequency_Hz: float
    r_pu: float
    x_pu: float
    tags: ClassVar[list[str]] = ["branch"]
    pf: Optional[PowerFlowVariables] = None
    emt_init: Optional[InitialConditionsEMT] = None
    type: str = "se_rl"
    ssm: Optional[StateSpaceModel] = None
    variables_emt: Optional[VariablesEMT] = None
    id_variables_emt: Optional[dict] = None
    from_bus_id: int = None
    to_bus_id: int = None

    def post_system_init(self, system):
        self.from_bus_id = next((n for n in system.bus if n.name == self.from_bus)).id
        self.to_bus_id = next((n for n in system.bus if n.name == self.to_bus)).id

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.branches.loc[f"{self.type}_{self.id}"]
        self.pf = PowerFlowVariables(
            vmag_from_bus=sol.from_bus_vmag.item(),
            vphase_from_bus=sol.from_bus_vphase.item(),
            vmag_to_bus=sol.to_bus_vmag.item(),
            vphase_to_bus=sol.to_bus_vphase.item(),
        )

    def load_ac_power_flow_solution(self, timepoint: str, pf_solution: ACPowerFlowSolution):
        self.pf = PowerFlowVariables(
            vmag_from_bus=pf_solution.bus_voltage_magnitude[self.from_bus_id, timepoint],
            vphase_from_bus=pf_solution.bus_voltage_angle[self.from_bus_id, timepoint],
            vmag_to_bus=pf_solution.bus_voltage_magnitude[self.to_bus_id, timepoint],
            vphase_to_bus=pf_solution.bus_voltage_angle[self.to_bus_id, timepoint],
        )
        print('ok')
        
    def _calculate_emt_initial_conditions(self):
        
        r = self.r_pu
        x = self.x_pu

        vmag_from_bus = self.pf.vmag_from_bus
        vphase_from_bus = self.pf.vphase_from_bus

        vmag_to_bus = self.pf.vmag_to_bus
        vphase_to_bus = self.pf.vphase_to_bus

        v_from_bus_DQ = vmag_from_bus * np.exp(vphase_from_bus * np.pi / 180 * 1j)
        v_to_bus_DQ = vmag_to_bus * np.exp(vphase_to_bus * np.pi / 180 * 1j)

        i_br_DQ = (v_from_bus_DQ - v_to_bus_DQ) / (r + 1j * x)

        self.emt_init = InitialConditionsEMT(
            vmag_from_bus=vmag_from_bus,
            vphase_from_bus=vphase_from_bus,
            vmag_to_bus=vmag_to_bus,
            vphase_to_bus=vphase_to_bus,
            v_from_bus_D=v_from_bus_DQ.real,
            v_from_bus_Q=v_from_bus_DQ.imag,
            v_to_bus_D=v_to_bus_DQ.real,
            v_to_bus_Q=v_to_bus_DQ.imag,
            i_br_D=i_br_DQ.real,
            i_br_Q=i_br_DQ.imag,
        )

    def _build_small_signal_model(self):

        rse = self.r_pu
        xse = self.x_pu
        wb = 2 * np.pi * self.base_frequency_Hz

        # Define state-space matrices (turn off code formatters for matrices)
        # fmt: off
        A = wb * np.array(
            [[-rse/xse,        1], 
             [      -1, -rse/xse]])

        B = wb * np.array(
            [[1/xse,     0, -1/xse,      0], 
             [    0, 1/xse,      0, -1/xse]])
        # fmt: on
        C = np.eye(2)

        D = np.zeros((2, 4))

        u = DynamicalVariables(
            name=["v_from_bus_D", "v_from_bus_Q", "v_to_bus_D", "v_to_bus_Q"],
            component=f"se_rl_{self.id}",
            type=["grid", "grid", "grid", "grid"],
            init=[
                self.emt_init.v_from_bus_D,
                self.emt_init.v_from_bus_Q,
                self.emt_init.v_to_bus_D,
                self.emt_init.v_to_bus_Q,
            ],
        )

        x = DynamicalVariables(
            name=["i_br_D", "i_br_Q"],
            component=f"se_rl_{self.id}",
            init=[self.emt_init.i_br_D, self.emt_init.i_br_Q],
        )
        y = copy.deepcopy(x)

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)

    def define_variables_emt(self):

        # States
        # ------
        i_br_D, i_br_Q = self.emt_init.i_br_D, self.emt_init.i_br_Q
        i_br_a, i_br_b, i_br_c = dq02abc(i_br_D, i_br_Q, 0, 0)

        x = DynamicalVariables(
            name=["i_br_a", "i_br_b", "i_br_c"],
            component=f"{self.type}_{self.id}",
            init=[i_br_a, i_br_b, i_br_c],
        )

        # Inputs
        u = DynamicalVariables(
            name=["v_from_bus_a", "v_from_bus_b", "v_from_bus_c", 
                  "v_to_bus_a", "v_to_bus_b", "v_to_bus_c"],
            component=f"{self.type}_{self.id}",
            type=["grid", "grid", "grid", "grid", "grid", "grid"],
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_br_a", "i_br_b", "i_br_c"],
            component=f"{self.type}_{self.id}",
        )

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)

    def get_derivative_state_emt(self):

        # Get state values
        i_br_a, i_br_b, i_br_c = self.variables_emt.x.value

        # Get input values
        v_from_bus_a, v_from_bus_b, v_from_bus_c, v_to_bus_a, v_to_bus_b, v_to_bus_c = self.variables_emt.u.value

        # Get parameters
        r = self.r_pu
        x = self.x_pu
        wb = 2 * np.pi * self.base_frequency_Hz

        # Differential equations
        d_i_br_a = wb / x   * (v_from_bus_a - v_to_bus_a - r * i_br_a)
        d_i_br_b = wb / x * (v_from_bus_b - v_to_bus_b - r * i_br_b)
        d_i_br_c = wb / x * (v_from_bus_c - v_to_bus_c - r * i_br_c)

        return [d_i_br_a, d_i_br_b, d_i_br_c]
        

    def get_output_emt(self):

        i_br_a, i_br_b, i_br_c = self.variables_emt.x.value

        return [i_br_a, i_br_b, i_br_c]
    
    def plot_results_emt(self, output_dir):
        
        # Retrieve simulation results
        time = self.variables_emt.x.time
        angle_ref =  2 * np.pi * self.base_frequency_Hz * time
        i_br_a, i_br_b, i_br_c = self.variables_emt.x.value
       
        # Transform abc to dq0
        i_br_D, i_br_Q, _ = zip(*map(abc2dq0, i_br_a, i_br_b, i_br_c, angle_ref))

        # Plot results
        fig = make_subplots(rows=1, cols=2)

        fig.add_trace(go.Scatter(x=time, y=i_br_D), row=1, col=1)
        fig.update_xaxes(title_text='Time [s]', row=1, col=1)
        fig.update_yaxes(title_text='i_br_D [p.u.]', row=1, col=1)

        fig.add_trace(go.Scatter(x=time, y=i_br_Q), row=1, col=2)
        fig.update_xaxes(title_text='Time [s]', row=1, col=2)
        fig.update_yaxes(title_text='i_br_Q [p.u.]', row=1, col=2)

        name = f"{self.type}_{self.id}"
        fig.update_layout(  title_text = name,
                            title_x=0.5,
                            showlegend = False,
                            )

        fig.write_html(os.path.join(output_dir, name + ".html"))


        