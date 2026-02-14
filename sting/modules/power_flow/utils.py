# ----------------------
# Import python packages
# ----------------------
from typing import NamedTuple


# -----------
# Sub-classes 
# -----------
class ModelSettings(NamedTuple):
    generator_type_costs: str = "linear"
    power_flow_formulation: str = "polar"
    load_shedding: bool = True
    write_model_file: bool = False

class SolverSettings(NamedTuple):
    """
    Settings for the solver for the capacity expansion model.
    """
    solver_name: str = "ipopt"
    tee: bool = True
    solver_options: dict = None

class ACPowerFlowSolution(NamedTuple):
    generator_active_dispatch: dict
    generator_reactive_dispatch: dict
    bus_voltage_magnitude: dict
    bus_voltage_angle: dict