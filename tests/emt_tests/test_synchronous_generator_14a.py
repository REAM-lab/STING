import os

import matplotlib
import numpy as np
import polars as pl
import pylab as plt
from scipy.integrate import solve_ivp

from sting import datasets, main
from sting.generator import SynchronousGenerator14A

from sting.generator.core import PowerFlowVariables
from sting.utils.dynamical_systems import make_smooth_step
from sting.utils.transformations import dq02abc, abc2dq0

from sting.utils.plotting_tools import compare_timeseries

matplotlib.use('TkAgg')

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)


gen = SynchronousGenerator14A(
    bus="bus_2", name="gen1",
    # Power flow 
    minimum_active_power_MW=40, maximum_active_power_MW=50, minimum_reactive_power_MVAR=-10, maximum_reactive_power_MVAR=10,
    cost_variable_USDperMWh=10,
    # Per unit system
    base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # Shaft, governor and turbine parameters Kundur (page 598)
    h_s=2, kd_w_pu=100,                       # Shaft
    kr_pu=0.05, tau_g_s=0.2,                 # Governor
    # Machine parameters Kundur (page 155)
    x_d_pu=1.81, x_q_pu=1.76, x_l_pu = 0.15, r_a_pu=0.003, 
    x_td_pu=0.3, x_tq_pu=0.65, x_std_pu=0.23, x_stq_pu=0.25,
    t_td0_s=8.0, t_tq0_s=1, t_std0_s=0.03, t_stq0_s=0.07,
    x_0_pu=0.25,
    # Shunt
    csh_pu=0.066, rsh_pu=15,
    # Branch and transformer
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, 
    txr_r1_pu=0.01, txr_x1_pu=0.1, txr_r2_pu=0.02, txr_x2_pu=0.1, 
)

system = datasets.toy_2(case_directory=case_directory)
system.add(gen)
system.apply("post_system_init", system)

inputs = {
    "synchronous_generator_14a_0": {
        "p_ref": lambda t: -0.2 if t > 0.1 else 0,
        "v_ref": lambda t: 0.2 if t > 1.1 else 0
    }
}
t_max=2.5

# EMT
main.run_emt(t_max, inputs, case_directory, system=system)
# SSM
_, ssm = main.run_ssm(case_directory, system=system)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# QBM 
_, qbm = main.run_qbm(case_directory, system=system)
sol = qbm.simulate(t_max=t_max, inputs=inputs)
qbm.write_simulation_csv(sol, os.path.join(case_directory, "outputs", "quadratic_bilinear"))

# Compare the results of the EMT and small-signal model simulations
file = "synchronous_generator_14a_0.csv"
cols_emt =["w", "governor", "i_stator_d", "i_stator_q", "i_field_d", "i_damper_1d", "i_damper_1q", "i_damper_2q", "v_shunt_D", "v_shunt_Q", "i_bus_D", "i_bus_Q"]
cols_ssm = ["w", "x_gov", "i_d", "i_q", "i_fd", "i_1d", "i_1q", "i_2q", "v_sh_D", "v_sh_Q", "i_br_D", "i_br_Q"]
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



# Compare the results of the EMT and small-signal model simulations
compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/simulation_emt/{file}"),
    df2=pl.read_csv(f"{case_directory}/outputs/quadratic_bilinear/{file}"),
    left_to_right=dict(zip(cols_emt, cols_ssm)),
    df1_name="EMT",
    df2_name="QBM",
    figure_filepath=f"{case_directory}/outputs/comparison_plot_qbm.html",
    df1_color="blue",
    df2_color="red"
)



print("ok")