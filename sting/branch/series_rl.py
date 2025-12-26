from dataclasses import dataclass, field
from sting.utils.transformations import dq02abc, abc2dq0
from typing import NamedTuple, Optional, ClassVar
import numpy as np
import copy

from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables


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
    idx: int = field(default=-1, init=False)
    from_bus: int
    to_bus: int
    sbase: float
    vbase: float
    fbase: float
    r: float
    l: float
    name: str = field(default_factory=str)
    tags: ClassVar[list[str]] = ["branch"]
    pf: Optional[PowerFlowVariables] = None
    emt_init: Optional[InitialConditionsEMT] = None
    type: str = "se_rl"
    ssm: Optional[StateSpaceModel] = None
    var_emt: Optional[VariablesEMT] = None


    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.branches.loc[f"se_rl_{self.idx}"]
        self.pf = PowerFlowVariables(
            vmag_from_bus=sol.from_bus_vmag.item(),
            vphase_from_bus=sol.from_bus_vphase.item(),
            vmag_to_bus=sol.to_bus_vmag.item(),
            vphase_to_bus=sol.to_bus_vphase.item(),
        )

    def _calculate_emt_initial_conditions(self):
        vmag_from_bus = self.pf.vmag_from_bus
        vphase_from_bus = self.pf.vphase_from_bus

        vmag_to_bus = self.pf.vmag_to_bus
        vphase_to_bus = self.pf.vphase_to_bus

        v_from_bus_DQ = vmag_from_bus * np.exp(vphase_from_bus * np.pi / 180 * 1j)
        v_to_bus_DQ = vmag_to_bus * np.exp(vphase_to_bus * np.pi / 180 * 1j)

        i_br_DQ = (v_from_bus_DQ - v_to_bus_DQ) / (self.r + 1j * self.l)

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

        rse = self.r
        lse = self.l
        wb = 2 * np.pi * self.fbase

        # Define state-space matrices (turn off code formatters for matrices)
        # fmt: off
        A = wb * np.array(
            [[-rse/lse,        1], 
             [      -1, -rse/lse]])

        B = wb * np.array(
            [[1/lse,     0, -1/lse,      0], 
             [    0, 1/lse,      0, -1/lse]])
        # fmt: on
        C = np.eye(2)

        D = np.zeros((2, 4))

        u = DynamicalVariables(
            name=["v_from_bus_D", "v_from_bus_Q", "v_to_bus_D", "v_to_bus_D"],
            component=f"se_rl_{self.idx}",
            type="grid",
            init=[
                self.emt_init.v_from_bus_D,
                self.emt_init.v_from_bus_Q,
                self.emt_init.v_to_bus_D,
                self.emt_init.v_to_bus_Q,
            ],
        )

        x = DynamicalVariables(
            name=["i_br_D", "i_br_Q"],
            component=f"se_rl_{self.idx}",
            type="grid",
            init=[self.emt_init.i_br_D, self.emt_init.i_br_Q],
        )
        y = copy.deepcopy(x)

        self.ssm = StateSpaceModel(A=A, B=B, C=C, D=D, u=u, y=y, x=x)

    def _define_variables_emt(self, **kwargs):

        # States
        # ------
        angle_init = kwargs.get("angle_init", 0.0)
        i_br_D, i_br_Q = self.emt_init.i_br_D, self.emt_init.i_br_Q
        i_br_a, i_br_b, i_br_c = dq02abc(i_br_D, i_br_Q, 0, angle_init)

        x = DynamicalVariables(
            name=["i_br_a", "i_br_b", "i_br_c"],
            component=f"{self.type}_{self.idx}",
            init=[i_br_a, i_br_b, i_br_c],
        )

        # Inputs
        u = DynamicalVariables(
            name=["v_from_bus_a", "v_from_bus_b", "v_from_bus_c", 
                  "v_to_bus_a", "v_to_bus_b", "v_to_bus_c"],
            component=f"{self.type}_{self.idx}",
            type=["grid", "grid", "grid", "grid", "grid", "grid"],
        )

        # Outputs
        y = DynamicalVariables(
            name=["i_br_a", "i_br_b", "i_br_c"],
            component=f"{self.type}_{self.idx}",
        )

        self.var_emt = VariablesEMT(x=x, u=u, y=y)

    def _get_derivative_state_emt(self, t, x, ud, ug, angle_sys):

        i_br_a, i_br_b, i_br_c = x

        v_from_bus_a, v_from_bus_b, v_from_bus_c, v_to_bus_a, v_to_bus_b, v_to_bus_c = ug

        r = self.r
        l = self.l
        wb = 2 * np.pi * self.fbase

        d_i_br_a = wb / l * (v_from_bus_a - v_to_bus_a - r * i_br_a)
        d_i_br_b = wb / l * (v_from_bus_b - v_to_bus_b - r * i_br_b)
        d_i_br_c = wb / l * (v_from_bus_c - v_to_bus_c - r * i_br_c)

        return [d_i_br_a, d_i_br_b, d_i_br_c]
        

    
    def _get_output_emt(self, t, x_vals, ud):
        
        i_br_a, i_br_b, i_br_c = x_vals

        return [i_br_a, i_br_b, i_br_c]