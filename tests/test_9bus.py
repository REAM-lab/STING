# -------------------------------------------------------
# Import libraries and components
# -------------------------------------------------------
import os
from sting import main
from sting.system import System
from sting.generator import InfiniteSource, GFMI18A, GFLI13A
from sting.line import LinePiModel
from sting.bus import Bus, Load
from sting.timescales import Timepoint
from sting.utils.dynamical_systems import smooth_step
from sting.utils.tuning import line_ieeerts79

# -------------------------------------------------------
# Directory for temporary files
# -------------------------------------------------------
case_directory = os.path.join(os.getcwd(), "tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# -------------------------------------------------------
# Definition of the 9-bus System
# -------------------------------------------------------

# Timepoint
t1 = Timepoint(name="t1", weight=1)

# Buses
bus_1 = Bus(name="bus_1", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.05)
bus_2 = Bus(name="bus_2", zone=None, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.05)
bus_3 = Bus(name="bus_3", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.05)
bus_4 = Bus(name="bus_4", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.05)
bus_5 = Bus(name="bus_5", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.05)
bus_6 = Bus(name="bus_6", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.05)
bus_7 = Bus(name="bus_7", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.05)
bus_8 = Bus(name="bus_8", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.95, maximum_voltage_pu=1.05)
bus_9 = Bus(name="bus_9", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, minimum_voltage_pu=0.90, maximum_voltage_pu=1.5)

# Loads
load_1 = Load(bus="bus_1", timepoint="t1", load_MW=0, load_MVAR=0)

# Transmission Lines
line_1_4 = LinePiModel(
    name="line_1_4", from_bus="bus_1", to_bus="bus_4", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, r_pu=0, x_pu=0.0576, g_pu=0, b_pu=0)
line_4_5 = LinePiModel(
    name="line_4_5", from_bus="bus_4", to_bus="bus_5", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, r_pu=0.017, x_pu=0.092, g_pu=0, b_pu=0.158)
line_5_6 = LinePiModel(
    name="line_5_6", from_bus="bus_5", to_bus="bus_6", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, r_pu=0.039, x_pu=0.17, g_pu=0, b_pu=0.358)
line_3_6 = LinePiModel(
    name="line_3_6", from_bus="bus_3", to_bus="bus_6", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, r_pu=0, x_pu=0.0586, g_pu=0, b_pu=0)
line_6_7 = LinePiModel(
    name="line_6_7", from_bus="bus_6", to_bus="bus_7", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, r_pu=0.0119, x_pu=0.1008, g_pu=0, b_pu=0.209)
line_7_8 = LinePiModel(
    name="line_7_8", from_bus="bus_7", to_bus="bus_8", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, r_pu=0.0085, x_pu=0.072, g_pu=0, b_pu=0.149)
line_8_2 = LinePiModel(
    name="line_8_2", from_bus="bus_8", to_bus="bus_2", zone=None, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, r_pu=0, x_pu=0.0625, g_pu=0, b_pu=0)
line_8_9 = LinePiModel(
    name="line_8_9", from_bus="bus_8", to_bus="bus_9", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, r_pu=0.032, x_pu=0.161, g_pu=0, b_pu=0.306)
line_9_4 = LinePiModel(
    name="line_9_4", from_bus="bus_9", to_bus="bus_4", zone="external", base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60, r_pu=0.01, x_pu=0.085, g_pu=0, b_pu=0.176)

# Add resistance and susceptance to lines based on typical values per mile for 230 kV lines to avoid zero values. The typical values are based on the IEEE RTS-79 test system.
typical_parameters_per_mile = line_ieeerts79(base_voltage_kv=230, miles=1) 
r_pu_mile = typical_parameters_per_mile["r_pu"]
x_pu_mile = typical_parameters_per_mile["x_pu"]
b_pu_mile = typical_parameters_per_mile["b_pu"]

for line in [line_1_4, line_4_5, line_5_6, line_3_6, line_6_7, line_7_8, line_8_2, line_8_9, line_9_4]:

    estimated_miles = line.x_pu / x_pu_mile
    if line.r_pu == 0:
        line.r_pu = r_pu_mile * estimated_miles
        print(f"Estimated r_pu for {line.name}: {line.r_pu:.6f}")

    if line.b_pu == 0:
        line.b_pu = b_pu_mile * estimated_miles
        print(f"Estimated b_pu for {line.name}: {line.b_pu:.6f}")

    if line.b_pu > 0:
        line.g_pu = line.b_pu * 0.01
        print(f"Estimated g_pu for {line.name}: {line.g_pu:.6f}")

# Print all lines
for line in [line_1_4, line_4_5, line_5_6, line_3_6, line_6_7, line_7_8, line_8_2, line_8_9, line_9_4]:
    print(f"{line.name}: r_pu={line.r_pu:.6f}, x_pu={line.x_pu:.6f}, g_pu={line.g_pu:.6f}, b_pu={line.b_pu:.6f}")

# Generation
source = InfiniteSource(
    name="grid", bus="bus_1", zone="external",
    minimum_active_power_MW=-200, maximum_active_power_MW=200, minimum_reactive_power_MVAR=-500, maximum_reactive_power_MVAR=500,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.001, x_pu=0.005
)

gfmi_1 = GFMI18A(
    name="gfmi_1", bus="bus_2", zone=None,
    # Power flow 
    minimum_active_power_MW=30, maximum_active_power_MW=50, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.005, xf1_pu=0.15, csh_pu=0.066, rsh_pu=1,
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

gfmi_2 = GFMI18A(
    name="gfmi_2", bus="bus_3", zone="external",
    # Power flow 
    minimum_active_power_MW=30, maximum_active_power_MW=70, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.005, xf1_pu=0.15, csh_pu=0.066, rsh_pu=1,
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


gfli_1 = GFLI13A(
    name="gfli_1", bus="bus_5", zone="external",
    # Power flow 
    minimum_active_power_MW=-100, maximum_active_power_MW=-50, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.002, xf1_pu=0.07, csh_pu=0.01, rsh_pu=1, 
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.003/2, txr_x1_pu=0.08/2, txr_r2_pu=0.003/2, txr_x2_pu=0.08/2, 
    # Phase-locked loop (PLL)
    kp_pll_pu=100, ki_pll_puHz=2500, tau_pll=1/100,
    # Inner current controller
    kp_cc_pu=0.05, ki_cc_puHz=0.6, kff_cc=0.75,
    # Power controllers
    kp_pc_pu=0.01, ki_pc_puHz=0.6
)

system = System(case_directory=case_directory)
buses = [bus_1, bus_2, bus_3, bus_4, bus_5, bus_6, bus_7, bus_8, bus_9]
timepoints = [t1]
loads = [load_1]
lines = [line_1_4, line_4_5, line_5_6, line_3_6, line_6_7, line_7_8, line_8_2, line_8_9, line_9_4]
generators = [source, gfmi_1, gfmi_2, gfli_1]

# Build grid model
for component in buses + timepoints + loads + lines + generators:
    system.add(component)

system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulation
# -------------------------------------------------------

# Step function inputs to simulate
def step2(t):
    return -0.05 if t >= 100 else 0.0

inputs = {
    'infinite_sources_0': {
        'v_ref_d': lambda t: 0
        }, 
    'gfmi_18a_0': {
        'p_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=0.10, transient_width=5e-3),
        'q_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=-0.10, transient_width=5e-3)
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


