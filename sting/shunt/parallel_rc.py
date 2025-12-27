import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, ClassVar
from sting.utils.transformations import dq02abc, abc2dq0
import copy

from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables


class PowerFlowVariables(NamedTuple):
    vmag_bus: float
    vphase_bus: float


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
class ShuntParallelRC:
    idx: int = field(default=-1, init=False)
    bus_idx: int
    sbase: float
    vbase: float
    fbase: float
    r: float
    c: float
    pf: Optional[PowerFlowVariables] = None
    emt_init: Optional[InitialConditionsEMT] = None
    ssm: Optional[StateSpaceModel] = None
    name: str = field(default_factory=str)
    type: str = "pa_rc"
    tags: ClassVar[list[str]] = ["shunt"]
    var_emt: Optional[VariablesEMT] = None
    idx_solve_ivp: Optional[dict] = None

    @property
    def g(self):
        return 1 / self.r

    @property
    def b(self):
        return 1 / self.c

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.shunts.loc[f"pa_rc_{self.idx}"]
        self.pf = PowerFlowVariables(
            vmag_bus=sol.bus_vmag.item(), vphase_bus=sol.bus_vphase.item()
        )

    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.pf.vmag_bus
        vphase_bus = self.pf.vphase_bus

        v_bus_DQ = vmag_bus * np.exp(vphase_bus * 1j * np.pi / 180)
        i_bus_DQ = v_bus_DQ * self.g + v_bus_DQ * (1j * self.b)

        self.emt_init = InitialConditionsEMT(
            vmag_bus=vmag_bus,
            vphase_bus=vphase_bus,
            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag,
            i_bus_D=i_bus_DQ.real,
            i_bus_Q=i_bus_DQ.imag,
        )

    def _build_small_signal_model(self):
        g = self.g
        b = self.b
        wb = 2 * np.pi * self.fbase

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
            component=f"pa_rc_{self.idx}",
            type="grid",
            init=[self.emt_init.i_bus_D, self.emt_init.i_bus_Q],
        )

        x = DynamicalVariables(
            name=["v_bus_D", "v_bus_Q"],
            component=f"pa_rc_{self.idx}",
            type="grid",
            init=[self.emt_init.v_bus_D, self.emt_init.v_bus_Q],
        )
        y = copy.deepcopy(x)

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)

    def _define_variables_emt(self, **kwargs):

        # States
        # ------
        angle_init = kwargs.get("angle_init", 0.0)
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, angle_init)

        x = DynamicalVariables(
            name=["v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type}_{self.idx}",
            init=[v_bus_a, v_bus_b, v_bus_c],
        )

        # Inputs
        u = DynamicalVariables(
            name=["i_bus_a", "i_bus_b", "i_bus_c"],
            component=f"{self.type}_{self.idx}",
            type=["grid", "grid", "grid"],
        )

        # Outputs
        y = DynamicalVariables(
            name=["v_bus_a", "v_bus_b", "v_bus_c"],
            component=f"{self.type}_{self.idx}",
        )

        self.var_emt = VariablesEMT(x=x, u=u, y=y)

    def _get_derivative_state_emt(self, t, x, ud, ug, angle_sys):

        v_bus_a, v_bus_b, v_bus_c = x

        g = self.g
        b = self.b
        wb = 2 * np.pi * self.fbase

        i_bus_a, i_bus_b, i_bus_c = ug

        d_v_bus_a = wb / b * (- g * v_bus_a + i_bus_a)
        d_v_bus_b = wb / b * (- g * v_bus_b + i_bus_b)
        d_v_bus_c = wb / b * (- g * v_bus_c + i_bus_c)

        return [d_v_bus_a, d_v_bus_b, d_v_bus_c]
    
    def _get_output_emt(self, t, x_vals, ud):
        v_bus_a, v_bus_b, v_bus_c = x_vals

        return [v_bus_a, v_bus_b, v_bus_c]