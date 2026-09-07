import logging
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

from sting.modules.power_flow.utils import ACPowerFlowSolution
from sting.system.component import Component
from sting.utils.dynamical_systems import (
    DynamicalVariables,
    QuadraticBilinearModel,
    StateSpaceModel,
)

# Set up logging
logger = logging.getLogger(__name__)


class PowerFlowVariables(NamedTuple):
    vmag_from_bus: float
    vphase_from_bus: float
    vmag_to_bus: float
    vphase_to_bus: float


class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables


@dataclass(slots=True, kw_only=True)
class Branch(Component):
    from_bus: str
    to_bus: str
    base_power_MVA: float
    base_voltage_kV: float
    base_frequency_Hz: float
    tags: ClassVar[list[str]] = ["ccm_branch"]
    power_flow_variables: PowerFlowVariables = None
    ssm: StateSpaceModel = None
    qbm: QuadraticBilinearModel = None
    variables_emt: VariablesEMT = None
    id_variables_emt: dict = None
    from_bus_id: int = None
    to_bus_id: int = None

    def post_system_init(self, system):
        self.from_bus_id = next((n for n in system.buses if n.name == self.from_bus)).id
        self.to_bus_id = next((n for n in system.buses if n.name == self.to_bus)).id

    def load_ac_power_flow_solution(
        self, timepoint: str, pf_solution: ACPowerFlowSolution
    ):
        self.power_flow_variables = PowerFlowVariables(
            vmag_from_bus=pf_solution.bus_voltage_magnitude[
                self.from_bus_id, timepoint
            ],
            vphase_from_bus=pf_solution.bus_voltage_angle[self.from_bus_id, timepoint],
            vmag_to_bus=pf_solution.bus_voltage_magnitude[self.to_bus_id, timepoint],
            vphase_to_bus=pf_solution.bus_voltage_angle[self.to_bus_id, timepoint],
        )
