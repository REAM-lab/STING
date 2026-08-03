import os

from sting.system import System

# Core components
from sting.generator import VoltageSource4A
from sting.line import LinePiModel
from sting.bus import Bus
from sting.load import Load
from sting.timescales import Timepoint

# -------------------------------------------------------
# Construct a simple 2-bus system
# -------------------------------------------------------
def toy_2(case_directory=None):
    """
    Create a simple "toy" 2 bus system with a stiff voltage source
    at bus 1 connected to a load at bus 2.
                 
                bus 1                  bus 2
                  ├─────VVVV────UUUU─────┤ 
                  │                      │    
    ┌─────────────┴──┐               ┌───┴──┐   
    │ Voltage Source │               │ Load │
    └────────────────┘               └──────┘
    """
    t1 = Timepoint(name="t1", weight=1)

    # Buses
    bus_1 = Bus(name="bus_1", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=1, maximum_voltage_pu=1)
    bus_2 = Bus(name="bus_2", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.3)
    load_1 = Load(bus="bus_2", timepoint="t1", load_MW=50, load_MVAR=20)

    # Transmission
    line_1 = LinePiModel(
        name="line_1_2", from_bus="bus_1", to_bus="bus_2",
        base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
        r_pu=0.012, x_pu=0.1, g_pu=0, b_pu=0.21
        )

    # Generation
    source = VoltageSource4A(
        name="external_grid", bus="bus_1", 
        minimum_active_power_MW=-200, maximum_active_power_MW=200, minimum_reactive_power_MVAR=-500, maximum_reactive_power_MVAR=500,
        cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
        r_pu=0.001, x_pu=0.005
    )


    system = System(case_directory=case_directory)

    # Build grid model
    for component in [bus_1, bus_2, load_1, line_1, source, t1]:
        system.add(component)

    return system