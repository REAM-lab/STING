import os

import polars as pl

from sting import datasets, main
from sting.generator import GFLI16A
from sting.utils.plotting_tools import compare_timeseries

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# -------------------------------------------------------
# Construct a simple 2-bus system
# -------------------------------------------------------
gfli_1 = GFLI16A(
    name="gfli_1", bus="bus_2",
    # Power flow 
    minimum_active_power_MW=-100, maximum_active_power_MW=-50, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.002, xf1_pu=0.07, csh_pu=0.01, rsh_pu=1, 
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.003/2, txr_x1_pu=0.08/2, txr_r2_pu=0.003/2, txr_x2_pu=0.08/2, 
    # Phase-locked loop (PLL)
    kp_pll_rad_s=100, ki_pll_rad2_s2=2500, tau_pll_s=1/100,
    # Inner current controller
    kp_cc_pu=0.05, ki_cc_puHz=0.6, kff_cc=0.75,
    # Power controllers
    kp_pc_pu=0.1, ki_pc_puHz=100
)

system = datasets.toy_2(case_directory=case_directory)
system.add(gfli_1)
system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

# Step function inputs to simulate
def step1(t):
    return 0.5 if t >= 0.5 else 0.0

def step2(t):
    return -0.5 if t >= 0.5 else 0.0

inputs = {
    'gfli_16a_0': {
        'p_ref': step1,
        'q_ref': step2,
        }
}

t_max = 1.5 # Simulation length in seconds

# EMT
main.run_emt(t_max, inputs, case_directory, system=system)
# SSM
_, ssm = main.run_ssm(case_directory, system=system)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# QBM 
_, qbm = main.run_qbm(case_directory, system=system)
sol = qbm.simulate(t_max=t_max, inputs=inputs)
os.makedirs(os.path.join(case_directory, "outputs", "quadratic_bilinear"), exist_ok=True)
qbm.write_simulation_csv(sol, os.path.join(case_directory, "outputs", "quadratic_bilinear"))
qbm.write_simulation_plots(sol, os.path.join(case_directory, "outputs", "quadratic_bilinear"))


# -------------------------------------------------------
# Compare the results of the EMT and small-signal model simulations
# -------------------------------------------------------
file = "gfli_16a_0.csv"
cols_emt =["v_pll_q", "z_pll", "z_apc", "z_rpc", "z_cc_d", "z_cc_q", "i_vsc_d", "i_vsc_q", "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q"]
cols_ssm = ["v_pll_q", "z_pll", "z_apc", "z_rpc", "z_cc_d", "z_cc_q", "i_vsc_d", "i_vsc_q", "v_sh_d", "v_sh_q", "i_bus_d", "i_bus_q"]
cols_qbm = ["v_pll_q", "z_pll", "z_apc", "z_rpc", "z_cc_d", "z_cc_q", "i_vsc_d", "i_vsc_q"]

compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/simulation_emt/{file}"),
    df2=pl.read_csv(f"{case_directory}/outputs/small_signal_model/{file}"),
    left_to_right=dict(zip(cols_emt, cols_ssm)),
    df1_name="EMT",
    df2_name="SSM",
    figure_filepath=f"{case_directory}/outputs/comparison_plot.html",
    df1_color="blue",
    df2_color="red"
)

compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/simulation_emt/{file}"),
    df2=pl.read_csv(f"{case_directory}/outputs/quadratic_bilinear/{file}"),
    left_to_right=dict(zip(cols_emt, cols_qbm)),
    df1_name="EMT",
    df2_name="QBM",
    figure_filepath=f"{case_directory}/outputs/comparison_plot_qbm.html",
    df1_color="blue",
    df2_color="red"
)

print("ok")
