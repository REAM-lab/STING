import os

from sting import main
from sting.system import System

# Core components
from sting.generator import InfiniteSource, GFLI10A
from sting.line import LinePiModel
from sting.bus import Bus, Load
from sting.timescales import Timepoint

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "tmpdir")
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
line_1 = LinePiModel(
    name="lima_to_santiago", from_bus="lima", to_bus="santiago",
    base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.01, x_pu=0.5, g_pu=0.05, b_pu=0.06666666666667
    )
# Generation
source = InfiniteSource(
    name="lima_source", bus="lima", 
    minimum_active_power_MW=-200, maximum_active_power_MW=200, minimum_reactive_power_MVAR=-500, maximum_reactive_power_MVAR=500,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.01, x_pu=0.5
)
gfli = GFLI10A(
    name="santiago_gfl", bus="santiago",
    # Power flow 
    minimum_active_power_MW=80, maximum_active_power_MW=80, minimum_reactive_power_MVAR=50, maximum_reactive_power_MVAR=51,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.02, xf1_pu=0.1, csh_pu=0.1, rsh_pu=1, 
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.01, txr_x1_pu=0.1, txr_r2_pu=0.02, txr_x2_pu=0.1, 
    # PLL and inner current controller
    kp_pll_pu=1, ki_pll_puHz=5, kp_cc_pu=1, ki_cc_puHz=5, kff_cc_pu=0.75
)

system = System(case_directory=case_directory)

# Build grid model
for component in [bus_1, bus_2, load_1, load_2, line_1, source, gfli, t1]:
    system.add(component)

system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

# Step function inputs to simulate
def step1(t):
    return 0.05 if t >= 0.25 else 0.0

def step2(t):
    return -0.05 if t >= 0.75 else 0.0

inputs = {
    'infinite_sources_0': {
        'v_ref_d': lambda t: 0
        }, 
    'gfli_10a_0': {
        'i_bus_d_ref': step1,
        'i_bus_q_ref': step2,
        }
}

t_max = 1.5 # Simulation length in seconds


# Construct system and small-signal model
_, ssm = main.run_ssm(system=system, case_directory=case_directory)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation
main.run_emt(inputs=inputs, t_max=t_max, system=system, case_directory=case_directory)

emt_dir = os.path.join(case_directory, "outputs", "simulation_emt")
ssm_dir = os.path.join(case_directory, "outputs", "small_signal_model")