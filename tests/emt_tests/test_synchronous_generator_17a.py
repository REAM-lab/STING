import os

import matplotlib
import numpy as np
import polars as pl
import pylab as plt
from scipy.integrate import solve_ivp

from sting import datasets, main
from sting.generator import SynchronousGenerator17A

from sting.generator.core import PowerFlowVariables
from sting.utils.dynamical_systems import make_smooth_step
from sting.utils.transformations import dq02abc, abc2dq0

matplotlib.use('TkAgg')

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)


gen = SynchronousGenerator17A(
    bus="bus_2", name="gen1",
    # Power flow 
    minimum_active_power_MW=-100, maximum_active_power_MW=-50, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=10,
    # Per unit system
    base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # Machine parameters Kundur (page 155)
    x_d_pu=1.81, x_q_pu=1.76, x_l_pu = 0.15, r_a_pu=0.003, 
    x_td_pu=0.3, x_tq_pu=0.65, x_std_pu=0.23, x_stq_pu=0.25,
    t_td0_s=8.0, t_tq0_s=1, t_std0_s=0.03, t_stq0_s=0.07,
    x_0_pu=0.25,
    # Excitor parameters Kundur (page 364)
    ka_pu=187, ta_s=0.89, te_s=1.15, kf_pu=0.058,
    tf_s=0.62, tb_s=0.06, tc_s=0.173, tau_v_s=0.05,
    ke_pu=1,
    # Shunt
    csh_pu=0.066, rsh_pu=10,
    # Branch and transformer
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, 
    txr_r1_pu=0.01, txr_x1_pu=0.1, txr_r2_pu=0.02, txr_x2_pu=0.1, 
)

system = datasets.toy_2(case_directory=case_directory)
system.add(gen)
system.apply("post_system_init", system)

inputs = {
    "synchronous_generator_17a_0": {
        "v_ref": lambda t: 0.1 if t > 0.1 else 0
    }
}
t_max=10

main.run_emt(t_max, inputs, case_directory, system=system)
