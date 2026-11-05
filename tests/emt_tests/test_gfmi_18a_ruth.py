import os

import polars as pl
import pylab as plt

from sting import datasets, main
from sting.generator import GFMI18A, GFMI18ALIM
from sting.utils.dynamical_systems import make_smooth_step
from sting.utils.plotting_tools import compare_timeseries

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# -------------------------------------------------------
# Construct a simple 2-bus system
# -------------------------------------------------------
gfmi = GFMI18A(
    name="gfmi_1", bus="bus_2",
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

system = datasets.toy_2(case_directory=case_directory)
system.add(gfmi)
system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

inputs = {
    'voltage_source_4a_0': {
        'v_ref_d': lambda t: 0
        }, 
    'gfmi_18a_0': {
        'v_ref': make_smooth_step(step_time=0.10, initial_value=0.0, final_value=0.50, transient_width=5e-3),
        'p_ref': make_smooth_step(step_time=0.10, initial_value=0.0, final_value=0.50, transient_width=5e-3),
        'q_ref': make_smooth_step(step_time=0.10, initial_value=0.0, final_value=-0.50, transient_width=5e-3) 
        }
}
inputs = {}
t_max = 1.5 # Simulation length in seconds

# EMT
main.run_emt(t_max, inputs, case_directory, system=system)
# SSM
_, ssm = main.run_ssm(case_directory, system=system)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)

# make comparison
compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/simulation_emt/gfmi_18a_0.csv"),
    df2=pl.read_csv(f"{case_directory}/outputs/small_signal_model/gfmi_18a_0.csv"),
    left_to_right={ "w": "w",
                    "q_f": "q_f",
                    "z_vc_d": "z_vc_d",
                    "z_vc_q": "z_vc_q",
                    "z_cc_d": "z_cc_d",
                    "z_cc_q": "z_cc_q",
                    "i_vsc_d": "i_vsc_d",
                    "i_vsc_q": "i_vsc_q",
                    "v_sh_d": "v_lcl_sh_d",
                    "v_sh_q": "v_lcl_sh_q",
                    "i_bus_d": "i_bus_d",
                    "i_bus_q": "i_bus_q"},
    df1_name="EMT",
    df2_name="SSM",
    figure_filepath=f"{case_directory}/outputs/gfmi_18_a_nolim_emt_v_ssm.html",
    df1_color="blue",
    df2_color="red"
)

# Set up a temporary directory used by all tests
case_directory2 = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir_lim")
os.makedirs(case_directory2, exist_ok=True)

# -------------------------------------------------------
# Construct a simple 2-bus system
# -------------------------------------------------------
gfmi = GFMI18ALIM(
    name="gfmi_1", bus="bus_2",
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
    k_q_pu=0.2, w_q_puHz=4000,
    Imax=1000
)

system = datasets.toy_2(case_directory=case_directory2)
system.add(gfmi)
system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

inputs = {
    'voltage_source_4a_0': {
        'v_ref_d': lambda t: 0
        }, 
    'gfmi_18a_lim_0': {
        'v_ref': make_smooth_step(step_time=0.10, initial_value=0.0, final_value=0.50, transient_width=5e-3),
        'p_ref': make_smooth_step(step_time=0.10, initial_value=0.0, final_value=0.50, transient_width=5e-3),
        'q_ref': make_smooth_step(step_time=0.10, initial_value=0.0, final_value=-0.50, transient_width=5e-3) 
        }
}
inputs = {}
t_max = 1.5 # Simulation length in seconds

# EMT
main.run_emt(t_max, inputs, case_directory2, system=system)
# SSM
_, ssm = main.run_ssm(case_directory2, system=system)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)


compare_timeseries(
    df1=pl.read_csv(f"{case_directory2}/outputs/simulation_emt/gfmi_18a_lim_0.csv"),
    df2=pl.read_csv(f"{case_directory2}/outputs/small_signal_model/gfmi_18a_lim_0.csv"),
    left_to_right={ "w": "w",
                    "q_f": "q_f",
                    "z_vc_d": "z_vc_d",
                    "z_vc_q": "z_vc_q",
                    "z_cc_d": "z_cc_d",
                    "z_cc_q": "z_cc_q",
                    "i_vsc_d": "i_vsc_d",
                    "i_vsc_q": "i_vsc_q",
                    "v_sh_d": "v_lcl_sh_d",
                    "v_sh_q": "v_lcl_sh_q",
                    "i_bus_d": "i_bus_d",
                    "i_bus_q": "i_bus_q"},
    df1_name="EMT",
    df2_name="SSM",
    figure_filepath=f"{case_directory}/outputs/gfmi_18_lim_emt_v_ssm.html",
    df1_color="blue",
    df2_color="red"
)


# Compare lim v no lim 

compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/simulation_emt/gfmi_18a_0.csv"),
    df2=pl.read_csv(f"{case_directory2}/outputs/simulation_emt/gfmi_18a_lim_0.csv"),
    left_to_right={ "w": "w",
                    "q_f": "q_f",
                    "z_vc_d": "z_vc_d",
                    "z_vc_q": "z_vc_q",
                    "z_cc_d": "z_cc_d",
                    "z_cc_q": "z_cc_q",
                    "i_vsc_d": "i_vsc_d",
                    "i_vsc_q": "i_vsc_q",
                    "v_sh_d": "v_sh_d",
                    "v_sh_q": "v_sh_q",
                    "i_bus_d": "i_bus_d",
                    "i_bus_q": "i_bus_q"},
    df1_name="GFMI18A",
    df2_name="GFMI18A Lim",
    figure_filepath=f"{case_directory}/outputs/gfmi_18_lim_v_nolim_emt.html",
    df1_color="blue",
    df2_color="red"
)


compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/small_signal_model/gfmi_18a_0.csv"),
    df2=pl.read_csv(f"{case_directory2}/outputs/small_signal_model/gfmi_18a_lim_0.csv"),
    left_to_right={ "w": "w",
                    "q_f": "q_f",
                    "z_vc_d": "z_vc_d",
                    "z_vc_q": "z_vc_q",
                    "z_cc_d": "z_cc_d",
                    "z_cc_q": "z_cc_q",
                    "i_vsc_d": "i_vsc_d",
                    "i_vsc_q": "i_vsc_q",
                    "v_lcl_sh_d": "v_lcl_sh_d",
                    "v_lcl_sh_q": "v_lcl_sh_q",
                    "i_bus_d": "i_bus_d",
                    "i_bus_q": "i_bus_q"},
    df1_name="GFMI18A",
    df2_name="GFMI18A Lim",
    figure_filepath=f"{case_directory}/outputs/gfmi_18_lim_v_nolim_ssm.html",
    df1_color="blue",
    df2_color="red"
)


