# ----------------------
# Import libraries
# ----------------------
from typing import NamedTuple
import os
import polars as pl

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

# -----------
# Functions
# -----------

def load_ac_power_flow_solution(directory: str) -> ACPowerFlowSolution:
    """
    Upload the solution of the optimization model back to the system object.

    Inputs:
    - directory: path to the directory where the solution files are stored
    - timepoint: name of the timepoint for which the solution is being loaded

    Outputs:
    - solution: ACPowerFlowSolution object containing the solution data
    """
    generator_dispatch = pl.read_csv(
            source=os.path.join(directory, 'generator_dispatch.csv'),
            schema_overrides={
                'id': pl.Int64,
                'type': pl.String,
                'timepoint': pl.String,
                'generator': pl.String, 
                'active_power_MW': pl.Float64, 
                'reactive_power_MVAR': pl.Float64
            }
        )
    bus_voltage = pl.read_csv(
            source=os.path.join(directory, 'bus_voltage.csv'),
            schema_overrides={
                'id': pl.Int64,
                'timepoint': pl.String,
                'bus': pl.String, 
                'voltage_magnitude_pu': pl.Float64, 
                'voltage_angle_deg': pl.Float64
            }
        )

    generator_keys = list(generator_dispatch.select(['id', 'timepoint', 'type']).iter_rows())
    active_generator_dispatch = dict( zip(generator_keys, generator_dispatch['active_power_MW']) )
    reactive_generator_dispatch = dict( zip(generator_keys, generator_dispatch['reactive_power_MVAR']) )

    bus_keys = list(bus_voltage.select(['id', 'timepoint']).iter_rows())
    bus_voltage_magnitude = dict( zip(bus_keys, bus_voltage['voltage_magnitude_pu']) )
    bus_voltage_angle = dict( zip(bus_keys, bus_voltage['voltage_angle_deg']) )
        
    solution = ACPowerFlowSolution(
            generator_active_dispatch=active_generator_dispatch,
            generator_reactive_dispatch=reactive_generator_dispatch,
            bus_voltage_magnitude=bus_voltage_magnitude,
            bus_voltage_angle=bus_voltage_angle)

    return solution        
