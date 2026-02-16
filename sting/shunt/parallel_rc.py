# -----------------------
# Import python packages
# -----------------------
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, ClassVar
from sting.utils.transformations import dq02abc, abc2dq0
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# -----------------------
# Import sting code
# -----------------------
from sting.shunt.core import Shunt
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables


class InitialConditionsEMT(NamedTuple):
    vmag_bus: float
    vphase_bus: float
    v_bus_D: float
    v_bus_Q: float
    i_bus_D: float
    i_bus_Q: float

class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables

@dataclass
class ShuntParallelRC(Shunt):
    #id: int = field(default=-1, init=False)
    #name: str
    #bus: str
    #base_power_MVA: float
    #base_voltage_kV: float
    #base_frequency_Hz: float
    g_pu: float
    b_pu: float
    bus_id: int = None
    #pf: Optional[PowerFlowVariables] = None
    emt_init: InitialConditionsEMT = None
    ssm: StateSpaceModel = None
    #type: str = "pa_rc"
    tags: ClassVar[list[str]] = ["shunt"]
    variables_emt: VariablesEMT = None
    id_variables_emt: dict = None

    def _calculate_emt_initial_conditions(self):
        
        g = self.g_pu
        b = self.b_pu

        vmag_bus = self.pf.vmag_bus
        vphase_bus = self.pf.vphase_bus

        v_bus_DQ = vmag_bus * np.exp(vphase_bus * 1j * np.pi / 180)
        i_bus_DQ = v_bus_DQ * g + v_bus_DQ * (1j * b)

        self.emt_init = InitialConditionsEMT(
            vmag_bus=vmag_bus,
            vphase_bus=vphase_bus,
            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag,
            i_bus_D=i_bus_DQ.real,
            i_bus_Q=i_bus_DQ.imag,
        )

    def _build_small_signal_model(self):
        g = self.g_pu
        b = self.b_pu
        wb = 2 * np.pi * self.base_frequency_Hz

        # Define state-space matrices (turn off code formatters for matrices)
        # fmt: off
        A = wb*np.array(
            [[-g/b,    1],
             [  -1, -g/b]])

        B = wb*np.array(
            [[1/b,   0],
             [  0, 1/b]])
        # fmt: on
        C = np.eye(2)

        D = np.zeros((2, 2))

        u = DynamicalVariables(
            name=["i_bus_D", "i_bus_Q"],
            component=f"pa_rc_{self.id}",
            type=["grid", "grid"],
            init=[self.emt_init.i_bus_D, self.emt_init.i_bus_Q],
        )

        x = DynamicalVariables(
            name=["v_bus_D", "v_bus_Q"],
            component=f"pa_rc_{self.id}",
            init=[self.emt_init.v_bus_D, self.emt_init.v_bus_Q],
        )
        y = copy.deepcopy(x)

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)

    def define_variables_emt(self):

        # States
        # ------
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, 0)

        x = DynamicalVariables(
            name=["v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type}_{self.id}",
            init=[v_bus_a, v_bus_b, v_bus_c],
        )

        # Inputs
        u = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type}_{self.id}",
            type=["grid", "grid", "grid"],
        )

        # Outputs
        y = DynamicalVariables(
            name=["v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type}_{self.id}",
        )

        self.variables_emt = VariablesEMT(x=x, u=u, y=y)

    def get_derivative_state_emt(self):

        # Get state values
        v_bus_a, v_bus_b, v_bus_c = self.variables_emt.x.value

        # Get input values
        i_bus_a, i_bus_b, i_bus_c = self.variables_emt.u.value

        # Get parameters
        g = self.g_pu
        b = self.b_pu
        wb = 2 * np.pi * self.base_frequency_Hz

        # Differential equations
        d_v_bus_a = wb / b * (- g * v_bus_a + i_bus_a)
        d_v_bus_b = wb / b * (- g * v_bus_b + i_bus_b)
        d_v_bus_c = wb / b * (- g * v_bus_c + i_bus_c)

        return [d_v_bus_a, d_v_bus_b, d_v_bus_c]
    
    def get_output_emt(self):

        v_bus_a, v_bus_b, v_bus_c = self.variables_emt.x.value

        return [v_bus_a, v_bus_b, v_bus_c]
    
    def plot_results_emt(self, output_dir):

        # Get state values
        v_bus_a, v_bus_b, v_bus_c = self.variables_emt.x.value
        time = self.variables_emt.x.time
        angle_ref =  2 * np.pi * self.fbase * time

        # Transform abc to dq0
        v_bus_D, v_bus_Q, _ = zip(*map(abc2dq0, v_bus_a, v_bus_b, v_bus_c, angle_ref))
        
        # Plot results
        fig = make_subplots(rows=1, cols=2)
        
        fig.add_trace(go.Scatter(x=time, y=v_bus_D), row=1, col=1)
        fig.update_xaxes(title_text='Time [s]', row=1, col=1)
        fig.update_yaxes(title_text='v_bus_D [p.u.]', row=1, col=1)

        fig.add_trace(go.Scatter(x=time, y=v_bus_Q), row=1, col=2)
        fig.update_xaxes(title_text='Time [s]', row=1, col=2)
        fig.update_yaxes(title_text='v_bus_Q [p.u.]', row=1, col=2)

        name = f"{self.type}_{self.id}"
        fig.update_layout(  title_text = name,
                            title_x=0.5,
                            showlegend = False,
                            )

        fig.write_html(os.path.join(output_dir, name + ".html"))
