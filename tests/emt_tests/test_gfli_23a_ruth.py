import os
import sys 
#sys.path.append("/Users/ruthkravis/Documents/STING")

from sting import main, datasets
from sting.system import System
import polars as pl

# Core components
from sting.generator import VoltageSource4A, GFLI23A
from sting.line import LinePiModel
from sting.bus import Bus
from sting.load import Load
from sting.timescales import Timepoint
from sting.utils.plotting_tools import compare_timeseries
from tests.emt_tests.participation_factor_helper import compute_participation_factors

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
    r_pu=0.0001, x_pu=0.001, g_pu=0.0005, b_pu=0.001
    )
# Generation
source = VoltageSource4A(
    name="lima_source", bus="lima", 
    minimum_active_power_MW=-200, maximum_active_power_MW=200, minimum_reactive_power_MVAR=-500, maximum_reactive_power_MVAR=500,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.001, x_pu=0.005
)

gfli = GFLI23A(
    name="santiago_gfl", bus="santiago",
    # Power flow 
    minimum_active_power_MW=80, maximum_active_power_MW=80, minimum_reactive_power_MVAR=50, maximum_reactive_power_MVAR=51,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.002, xf1_pu=0.07, csh_pu=0.01, rsh_pu=1, 
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.003/2, txr_x1_pu=0.08/2, txr_r2_pu=0.003/2, txr_x2_pu=0.08/2, 
    # Phase-locked loop (PLL)
    kp_pll_rad_s=100, ki_pll_rad2_s2=2500,
    # Inner current controller
    kp_cc_pu=0.05, ki_cc_puHz=0.6, kff_cc=0.75,
    # DC side
    kp_vdc_pu=1.2, ki_vdc_puHz=20, kp_iL_pu=1, ki_iL_puHz=10, 
    l_dc_pu = 0.1, c_dc_pu = 20,
    v_dc_ref = 2.0, v_s_pu = 0.5,
    Ti_L_s = 0.01, Tv_dc_s = 0.01, Ti_dc_s = 0.01, kff_idc = 1, kff_iload = 1, Ti_load_s = 0.01,
    r_eq_pu = 0.0001,
    v_eq_ref = 0.5, Cload=0.2, kv_p = 20.0, kv_i = 100.0, p_load = -0.8
)

system = System(case_directory=case_directory)

# Build grid modelx
for component in [bus_1, bus_2, load_1, line_1, source, gfli, t1]:
    system.add(component)

system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

# Step function inputs to simulate
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return -0.1 if t >= 0.5 else 0.0

inputs = {
    'gfli_23a_0': {
        'v_s': step1,
        'i_load_ref': step2,
        }
}
inputs = {}
t_max = 1.5 # Simulation length in seconds


# Construct system and small-signal model
_, ssm = main.run_ssm(system=system, case_directory=case_directory)
compute_participation_factors(output_dir=case_directory)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation
main.run_emt(inputs=inputs, t_max=t_max, system=system, case_directory=case_directory)

## Compare small signal and emt 


compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/simulation_emt/gfli_23a_0.csv"),
    df2=pl.read_csv(f"{case_directory}/outputs/small_signal_model/gfli_23a_0.csv"),
    left_to_right={ "z_cc_d": "z_cc_d",
                    "z_cc_q": "z_cc_q",
                    "v_sh_d": "v_lcl_sh_d",
                    "v_sh_q": "v_lcl_sh_q", 
                    "i_bus_d": "i_bus_d", 
                    "i_bus_q": "i_bus_q",
                    "i_vsc_d": "i_vsc_d",
                    "x_1": "x_1",
                    "x_2": "x_2",
                    "i_L_f": "i_L_f",
                    "v_dc_f": "v_dc_f",
                    "i_dc_f": "i_dc_f",
                    "i_load_f": "i_load_f",
                    "x_1": "x_1",
                    "x_2": "x_2",
                    "i_L": "i_L",
                    "v_dc": "v_dc",
                    "v_eq": "v_eq",
                    "gamma_load": "gamma_load"},
    df1_name="EMT",
    df2_name="SSM",
    figure_filepath=f"{case_directory}/outputs/gfli_23a_ssm_v_emt.html",
    df1_color="blue",
    df2_color="red"
)




# Set up a temporary directory used by all tests
case_directory2 = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir_lim")
os.makedirs(case_directory2, exist_ok=True)

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
    r_pu=0.0001, x_pu=0.001, g_pu=0.0005, b_pu=0.001
    )
# Generation
source = VoltageSource4A(
    name="lima_source", bus="lima", 
    minimum_active_power_MW=-200, maximum_active_power_MW=200, minimum_reactive_power_MVAR=-500, maximum_reactive_power_MVAR=500,
    cost_variable_USDperMWh=0, base_power_MVA=100, base_voltage_kV=230, base_frequency_Hz=60,
    r_pu=0.001, x_pu=0.005
)

gfli = GFLI23A(
    name="santiago_gfl", bus="santiago",
    # Power flow 
    minimum_active_power_MW=80, maximum_active_power_MW=80, minimum_reactive_power_MVAR=50, maximum_reactive_power_MVAR=51,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.002, xf1_pu=0.07, csh_pu=0.01, rsh_pu=1, 
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.003/2, txr_x1_pu=0.08/2, txr_r2_pu=0.003/2, txr_x2_pu=0.08/2, 
    # Phase-locked loop (PLL)
    kp_pll_rad_s=100, ki_pll_rad2_s2=2500,
    # Inner current controller
    kp_cc_pu=0.05, ki_cc_puHz=0.6, kff_cc=0.75,
    # DC side
    kp_vdc_pu=1.2, ki_vdc_puHz=20, kp_iL_pu=1, ki_iL_puHz=10, 
    l_dc_pu = 0.1, c_dc_pu = 20,
    v_dc_ref = 2.0, v_s_pu = 0.5,
    Ti_L_s = 0.01, Tv_dc_s = 0.01, Ti_dc_s = 0.01, kff_idc = 1, kff_iload = 1, Ti_load_s = 0.01,
    r_eq_pu = 0.0001,
    Imax = 1000, Imax_bat=1000, Emax = 1000, SOC_init=0.5, limiting=True,
    v_eq_ref = 0.5, Cload=0.2, kv_p = 20.0, kv_i = 100.0, p_load = -0.8
)

system = System(case_directory=case_directory2)

# Build grid modelx
for component in [bus_1, bus_2, load_1, line_1, source, gfli, t1]:
    system.add(component)

system.apply("post_system_init", system)


# Step function inputs to simulate
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return -0.1 if t >= 0.5 else 0.0

inputs = {
    'gfli_23a_0': {
        'v_s': step1,
        'i_load_ref': step2,
        }
}
inputs = {}
t_max = 1.5 # Simulation length in seconds


# Construct system and small-signal model
_, ssm = main.run_ssm(system=system, case_directory=case_directory2)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation
main.run_emt(inputs=inputs, t_max=t_max, system=system, case_directory=case_directory2)

compare_timeseries(
    df1=pl.read_csv(f"{case_directory2}/outputs/simulation_emt/gfli_23a_0.csv"),
    df2=pl.read_csv(f"{case_directory2}/outputs/small_signal_model/gfli_23a_0.csv"),
    left_to_right={ "z_cc_d": "z_cc_d",
                    "z_cc_q": "z_cc_q",
                    "v_sh_d": "v_lcl_sh_d",
                    "v_sh_q": "v_lcl_sh_q", 
                    "i_bus_d": "i_bus_d", 
                    "i_bus_q": "i_bus_q",
                    "i_vsc_d": "i_vsc_d",
                    "x_1": "x_1",
                    "x_2": "x_2",
                    "i_L_f": "i_L_f",
                    "v_dc_f": "v_dc_f",
                    "i_dc_f": "i_dc_f",
                    "i_load_f": "i_load_f",
                    "x_1": "x_1",
                    "x_2": "x_2",
                    "i_L": "i_L",
                    "v_dc": "v_dc",
                    "v_eq": "v_eq",
                    "gamma_load": "gamma_load"},
    df1_name="EMT",
    df2_name="SSM",
    figure_filepath=f"{case_directory}/outputs/gfli_23a_lim_ssm_v_emt.html",
    df1_color="blue",
    df2_color="red"
)


## Compare GFLI23A EMT to GFLI23LIM EMT 

# Compare EMT 
compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/simulation_emt/gfli_23a_0.csv"),
    df2=pl.read_csv(f"{case_directory2}/outputs/simulation_emt/gfli_23a_0.csv"),
    left_to_right={ "z_cc_d": "z_cc_d",
                    "z_cc_q": "z_cc_q",
                    "v_sh_d": "v_sh_d",
                    "v_sh_q": "v_sh_q", 
                    "i_bus_d": "i_bus_d", 
                    "i_bus_q": "i_bus_q",
                    "i_vsc_d": "i_vsc_d",
                    "x_1": "x_1",
                    "x_2": "x_2",
                    "i_L_f": "i_L_f",
                    "v_dc_f": "v_dc_f",
                    "i_dc_f": "i_dc_f",
                    "i_load_f": "i_load_f",
                    "x_1": "x_1",
                    "x_2": "x_2",
                    "i_L": "i_L",
                    "v_dc": "v_dc",
                    "v_eq": "v_eq",
                    "gamma_load": "gamma_load"},
    df1_name="GFL 23A",
    df2_name="GFL 23A LIM",
    figure_filepath=f"{case_directory2}/outputs/gfli_23a_lim_vs_nolim_emt.html",
    df1_color="blue",
    df2_color="red"
)

# Compare SSM 

compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/small_signal_model/gfli_23a_0.csv"),
    df2=pl.read_csv(f"{case_directory2}/outputs/small_signal_model/gfli_23a_0.csv"),
    left_to_right={ "z_cc_d": "z_cc_d",
                    "z_cc_q": "z_cc_q",
                    "v_lcl_sh_d": "v_lcl_sh_d",
                    "v_lcl_sh_q": "v_lcl_sh_q", 
                    "i_bus_d": "i_bus_d", 
                    "i_bus_q": "i_bus_q",
                    "i_vsc_d": "i_vsc_d",
                    "x_1": "x_1",
                    "x_2": "x_2",
                    "i_L_f": "i_L_f",
                    "v_dc_f": "v_dc_f",
                    "i_dc_f": "i_dc_f",
                    "i_load_f": "i_load_f",
                    "x_1": "x_1",
                    "x_2": "x_2",
                    "i_L": "i_L",
                    "v_dc": "v_dc",
                    "v_eq": "v_eq",
                    "gamma_load": "gamma_load"},
    df1_name="GFL 23A",
    df2_name="GFL 23A LIM",
    figure_filepath=f"{case_directory2}/outputs/gfli_23a_lim_vs_nolim_ssm.html",
    df1_color="blue",
    df2_color="red"
)
