import os
import numpy as np

from sting import main
from sting.system import System

# Core components
from sting.generator import UPS21A, VoltageSource4A
from sting.bus import Bus
from sting.line import LinePiModel
from sting.load import Load
from sting.timescales import Timepoint
from sting.modules.simulation_emt.core import SimulationEMT


case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# Construct the paper's single-data-center infinite-bus (SDCIB) system.
t1 = Timepoint(name="t1", weight=1)
bus_1 = Bus(
    name="infinite_bus",
    bus_type="slack",
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    minimum_voltage_pu=1,
    maximum_voltage_pu=1,
)
bus_2 = Bus(
    name="pcc",
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    minimum_voltage_pu=0.95,
    maximum_voltage_pu=1.3,
)
load_1 = Load(bus="pcc", timepoint="t1", load_MW=0, load_MVAR=0)
line_1 = LinePiModel(
    name="infinite_source_impedance",
    from_bus="infinite_bus",
    to_bus="pcc",
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    r_pu=0.01,
    x_pu=0.09,
    g_pu=0.00005,
    b_pu=0.0001,
)
source = VoltageSource4A(
    name="infinite_source",
    bus="infinite_bus",
    slack=True,
    minimum_active_power_MW=-200,
    maximum_active_power_MW=200,
    minimum_reactive_power_MVAR=-500,
    maximum_reactive_power_MVAR=500,
    cost_variable_USDperMWh=0,
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    r_pu=0.01,
    x_pu=0.1,
)
ups = UPS21A(
    name="santiago_ups",
    bus="pcc",
    minimum_active_power_MW=-50,
    maximum_active_power_MW=-50,
    minimum_reactive_power_MVAR=0,
    maximum_reactive_power_MVAR=0,
    cost_variable_USDperMWh=10,
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    r_afe=0.003,
    l_afe=0.05,
    c_dc=2.0,
    r_vsi=0.003,
    l_vsi=0.05,
    c_vsi=0.2,
    r_psu=0.005,
    c_psu=2.0,
    c_eq=0.2
)

system = System(case_directory=case_directory)
for component in [bus_1, bus_2, load_1, line_1, source, ups, t1]:
    system.add(component)
system.apply("post_system_init", system)


def load_step(t):
    return 0.05 if t >= 0.1 else 0.0


inputs = {'ups_21a_0': {'pload': load_step}}


_, ssm = main.run_ssm(system=system, case_directory=case_directory)

ssm.simulate_ssm(t_max=1.5, inputs=inputs)

main.run_emt(
    inputs=inputs,
    t_max=1.5,
    system=system,
    case_directory=case_directory,
)