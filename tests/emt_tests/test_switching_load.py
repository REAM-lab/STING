import os
import polars as pl
from sting import main
from sting.system import System
import pylab as plt
# Core components
from sting.generator import VoltageSource4A, GFMI18A
from sting.line import LinePiModel
from sting.bus import Bus
from sting.load import Load, SwitchingLoad
from sting.timescales import Timepoint
from sting.utils.dynamical_systems import smooth_step


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
load_1 = Load(bus="santiago", timepoint="t1", load_MW=50, load_MVAR=20)

# Transmission
line_1 = LinePiModel(
    name="lima_to_santiago", from_bus="lima", to_bus="santiago",
    base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.001, x_pu=0.01, g_pu=0.0005, b_pu=0.001
    )
# Generation
source = VoltageSource4A(
    name="lima_source", bus="lima", 
    minimum_active_power_MW=-200, maximum_active_power_MW=200, minimum_reactive_power_MVAR=-500, maximum_reactive_power_MVAR=500,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.001, x_pu=0.005
)
gfmi = GFMI18A(
    name="santiago_gfmi", bus="santiago",
    # Power flow 
    minimum_active_power_MW=80, maximum_active_power_MW=80, minimum_reactive_power_MVAR=50, maximum_reactive_power_MVAR=51,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.005, xf1_pu=0.15, csh_pu=0.066, rsh_pu=10,
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.01, txr_x1_pu=0.1, txr_r2_pu=0.02, txr_x2_pu=0.1, 
    # Inner voltage controller
    kp_vc_pu=0.562, ki_vc_puHz=484.989, kffi_vc=0.80,
    # Inner current controller
    kp_cc_pu=4.77, ki_cc_puHz=60, kffv_cc=0,
    # Virtual inertia
    h_s=2, kd_pu=70, 
    # Voltage droop
    k_q_pu=0.2, w_q_puHz=4000
)
# Switching load
switching_load = SwitchingLoad(
    name="santiago_switching_load", 
    bus="santiago",
    base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=1e-4, 
    x_pu=1e-6, # do not use zero to avoid singularity in EMT simulation
)

system = System(case_directory=case_directory)

# Build grid model
for component in [bus_1, bus_2, load_1, line_1, source, gfmi, t1, switching_load]:
    system.add(component)

system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

inputs = {
    'voltage_source_4a_0': {
        'v_ref_d': lambda t: 0
        }, 
    'switching_loads_0': {
        'connect': lambda t: True if ((t >= 0.1) and (t <= 0.12)) else False,
    }
}

t_max = 0.3 # Simulation length in seconds

# Run EMT simulation
main.run_emt(inputs=inputs, t_max=t_max, system=system, case_directory=case_directory)


