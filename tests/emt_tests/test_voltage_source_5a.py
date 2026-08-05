import os

from sting import main, datasets
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

# Generation
source_1 = VoltageSource5A(
    name="source_1", bus="bus_2", 
    minimum_active_power_MW=120, maximum_active_power_MW=140, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.001, x_pu=0.1, inertia_constant_s=1, damping_pu=1
)

source_2 = VoltageSource5A(
    name="source_2", bus="bus_3", 
    minimum_active_power_MW=200, maximum_active_power_MW=250, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.001, x_pu=0.1, inertia_constant_s=1, damping_pu=1
)
source_3 = VoltageSource5A(
    name="source_3", bus="bus_5", 
    minimum_active_power_MW=50, maximum_active_power_MW=100, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.001, x_pu=0.1, inertia_constant_s=1, damping_pu=1
)

system = datasets.wscc_9(case_directory=case_directory)
system.gfli_16a.clear()
system.gfmi_18a.clear()

# Build grid model
for component in [ source_1, source_2, source_3 ]:
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
