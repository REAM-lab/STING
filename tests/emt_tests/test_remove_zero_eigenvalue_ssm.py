import os

from plotly.subplots import make_subplots

from sting import datasets, main
from sting.generator import VoltageSource5A
from sting.load import Load
from sting.utils.plotting_tools import plot_eigenvalues

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# Generation
source_1 = VoltageSource5A(
    name="source_1",
    bus="bus_2",
    minimum_active_power_MW=80,
    maximum_active_power_MW=80,
    minimum_reactive_power_MVAR=50,
    maximum_reactive_power_MVAR=51,
    cost_variable_USDperMWh=0,
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    r_pu=0.001,
    x_pu=0.1,
    inertia_constant_s=1,
    damping_pu=1,
)

source_2 = VoltageSource5A(
    name="source_2",
    bus="bus_1",
    minimum_active_power_MW=-200,
    maximum_active_power_MW=200,
    minimum_reactive_power_MVAR=-500,
    maximum_reactive_power_MVAR=500,
    slack=True,
    cost_variable_USDperMWh=0,
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    r_pu=0.001,
    x_pu=0.1,
    inertia_constant_s=1,
    damping_pu=1,
)

load_1 = Load(bus="bus_2", timepoint="t1", load_MW=0, load_MVAR=0)

system = datasets.toy_2(case_directory=case_directory)

system.voltage_source_4a.clear()

# Build grid model
for component in [source_1, source_2, load_1]:
    system.add(component)

system.apply("post_system_init", system)


# -------------------------------------------------------
# Construct system and small-signal model
# -------------------------------------------------------

sys, ssm = main.run_ssm(system=system, case_directory=case_directory)

state_space_1 = ssm.model
state_space_2 = ssm.set_reference_phase_angle()

# -------------------------------------------------------
# Plot and compare eigenvalues
# -------------------------------------------------------

fig = make_subplots(rows=1, cols=1)
fig = plot_eigenvalues(fig, state_space_1.A)
fig = plot_eigenvalues(fig, state_space_2.A, marker_color="red", marker_symbol="triangle-up")

fig.write_html(os.path.join(case_directory, "eigenvalues.html"))