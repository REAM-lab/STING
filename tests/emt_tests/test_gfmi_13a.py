import os

import polars as pl
import pylab as plt

from sting import datasets, main
from sting.generator import GFMI13A
from sting.utils.dynamical_systems import make_smooth_step
from sting.utils.plotting_tools import compare_timeseries

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# -------------------------------------------------------
# Construct a simple 2-bus system
# -------------------------------------------------------
gfmi = GFMI13A(
    name="gfmi_1", bus="bus_2",
    # Power flow
    minimum_active_power_MW=80, maximum_active_power_MW=80, minimum_reactive_power_MVAR=50,maximum_reactive_power_MVAR=51, 
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.005, xf1_pu=0.15, csh_pu=0.066, rsh_pu=10,
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.01, txr_x1_pu=0.1, txr_r2_pu=0.02, txr_x2_pu=0.1,
    # Virtual inertia
    h_s=2, kd_pu=70,
    # Transient virtual resistor
    w_tvr_pu=60.0, R_v_pu=0.09,
)

system = datasets.toy_2(case_directory=case_directory)
system.add(gfmi)
system.apply("post_system_init", system)


# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

inputs = {
    "voltage_source_4a_0": {
        "v_ref_d": lambda t: 0
    },
    "gfmi_13a_0": {
        "v_ref": make_smooth_step(step_time=0.0010, initial_value=0.0, final_value=0.50, transient_width=5e-3),
        "p_ref": make_smooth_step(step_time=0.0010, initial_value=0.0, final_value=0.50, transient_width=5e-3),
    },
}
t_max = 1.5  # Simulation length in seconds

# EMT
main.run_emt(t_max, inputs, case_directory, system=system)
# SSM
_, ssm = main.run_ssm(case_directory, system=system)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

# ------------------------------------------------------------------
# Compare the results of the EMT and small-signal model simulations
# ------------------------------------------------------------------

file = "gfmi_13a_0.csv"
cols_emt = ["w", "z_tvr_d", "z_tvr_q", "i_vsc_d", "i_vsc_q", "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q"]
cols_ssm = ["w", "z_tvr_d", "z_tvr_q", "i_vsc_d", "i_vsc_q", "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q"]

compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/simulation_emt/{file}"),
    df2=pl.read_csv(f"{case_directory}/outputs/small_signal_model/{file}"),
    left_to_right=dict(zip(cols_emt, cols_ssm)),
    df1_name="EMT",
    df2_name="SSM",
    figure_filepath=(f"{case_directory}/outputs/comparison_plot.html"),
    df1_color="blue",
    df2_color="red",
)


print("ok")
