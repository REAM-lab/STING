import os

from sting import main
from sting.system import System

# Core components
from sting.generator import VoltageSource5A
from sting.line import LinePiModel
from sting.bus import Bus
from sting.load import Load
from sting.timescales import Timepoint

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# -------------------------------------------------------
# Construct a simple 2-bus system
# -------------------------------------------------------
t1 = Timepoint(name="t1", weight=1)
# Buses
bus_1 = Bus(name="lima", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=1, maximum_voltage_pu=1)
bus_2 = Bus(name="santiago", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.3)
load_1 = Load(bus="lima", timepoint="t1", load_MW=0, load_MVAR=0)
load_2 = Load(bus="santiago", timepoint="t1", load_MW=0, load_MVAR=0)
# Transmission
line = LinePiModel(
    name="lima_to_santiago", from_bus="lima", to_bus="santiago",
    base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.01, x_pu=0.5, g_pu=0.05, b_pu=0.06666666666667
    )
# Generation
source_1 = VoltageSource5A(
    name="lima_source", bus="lima", 
    minimum_active_power_MW=-200, maximum_active_power_MW=200, minimum_reactive_power_MVAR=-500, maximum_reactive_power_MVAR=500,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.01, x_pu=0.5, inertia_constant_s=3, damping_pu=1
)

source_2 = VoltageSource5A(
    name="santiago_source", bus="santiago", 
    minimum_active_power_MW=100, maximum_active_power_MW=100, minimum_reactive_power_MVAR=74, maximum_reactive_power_MVAR=75,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.05, x_pu=0.2, inertia_constant_s=3, damping_pu=1
)
system = System(case_directory=case_directory)

# Build grid model
for component in [bus_1, bus_2, load_1, load_2, line, source_1, source_2, t1]:
    system.add(component)

system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

# Step function inputs to simulate
def step1(t):
    return 0.01 if t >= 0.1 else 0.0

def step2(t):
    return 0.0

inputs = {
    'voltage_source_5a_0': {
        'p_m': step1
        }, 
    'voltage_source_5a_1': {
        'p_m': step2
        }
}

t_max = 1.0 # Simulation length in seconds


# Construct system and small-signal model
_, ssm = main.run_ssm(system=system, case_directory=case_directory)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation
main.run_emt(inputs=inputs, t_max=t_max, system=system, case_directory=case_directory)
