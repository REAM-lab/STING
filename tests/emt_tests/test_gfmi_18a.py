import os
import polars as pl
from sting import main
from sting.system import System
import pylab as plt
# Core components
from sting.generator import VoltageSource4A, GFMI18A
from sting.line import LinePiModel
from sting.bus import Bus
from sting.load import Load
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
#load_2 = Load(bus="santiago", timepoint="t1", load_MW=0, load_MVAR=0)
# Transmission
line_1 = LinePiModel(
    name="lima_to_santiago", from_bus="lima", to_bus="santiago",
    base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.0001, x_pu=0.001, g_pu=0.0005, b_pu=0.001
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

system = System(case_directory=case_directory)

# Build grid model
for component in [bus_1, bus_2, load_1, line_1, source, gfmi, t1]:
    system.add(component)

system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

# Step function inputs to simulate
def step2(t):
    return -0.05 if t >= 100 else 0.0

inputs = {
    'voltage_source_4a': {
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


# Compare emt vs ssm results
emt_results = pl.read_csv(f"{case_directory}/outputs/simulation_emt/gfmi_18a_0.csv")
# Take time, and psh
time_emt = emt_results.select("time").to_numpy().flatten()
p_sh_emt = emt_results.select("p_sh").to_numpy().flatten()
q_sh_emt = emt_results.select("q_sh").to_numpy().flatten()
v_sh_d_emt = emt_results.select("v_sh_d").to_numpy().flatten()
v_sh_q_emt = emt_results.select("v_sh_q").to_numpy().flatten()
i_bus_d_emt = emt_results.select("i_bus_d").to_numpy().flatten()
i_bus_q_emt = emt_results.select("i_bus_q").to_numpy().flatten

ssm_results = pl.read_csv(f"{case_directory}/outputs/small_signal_model/gfmi_18a_0.csv")
time_ssm = ssm_results.select("time").to_numpy().flatten()
v_sh_d_ssm = ssm_results.select("v_lcl_sh_d").to_numpy().flatten()
v_sh_q_ssm = ssm_results.select("v_lcl_sh_q").to_numpy().flatten()
i_bus_d_ssm = ssm_results.select("i_bus_d").to_numpy().flatten()
i_bus_q_ssm = ssm_results.select("i_bus_q").to_numpy().flatten()

# Compute power
p_sh_ssm = v_sh_d_ssm*i_bus_d_ssm + v_sh_q_ssm*i_bus_q_ssm
q_sh_ssm = v_sh_q_ssm*i_bus_d_ssm - v_sh_d_ssm*i_bus_q_ssm

# Plot:
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 6))
ax[0, 0].plot(time_emt, p_sh_emt, label="EMT")
ax[0, 0].plot(time_ssm, p_sh_ssm, label="SSM")
ax[0, 0].set_xlabel("Time [s]")
ax[0, 0].set_ylabel("Power [pu]")
ax[0, 0].legend() 

ax[0, 1].plot(time_emt, v_sh_d_emt, label="EMT")
ax[0, 1].plot(time_ssm, v_sh_d_ssm, label="SSM")
ax[0, 1].set_xlabel("Time [s]")
ax[0, 1].set_ylabel("Voltage d [pu]")
ax[0, 1].legend()

ax[1, 0].plot(time_emt, q_sh_emt, label="EMT")
ax[1, 0].plot(time_ssm, q_sh_ssm, label="SSM")
ax[1, 0].set_xlabel("Time [s]")
ax[1, 0].set_ylabel("Reactive Power [pu]")

ax[1, 1].plot(time_emt, i_bus_d_emt, label="SSM")
ax[1, 1].plot(time_emt, i_bus_d_ssm, label="EMT")
ax[1, 1].set_xlabel("Time [s]")
ax[1, 1].set_ylabel("Current d [pu]")


# Save the figure
plt.savefig(f"{case_directory}/outputs/comparison.png")
