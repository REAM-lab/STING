import os

import matplotlib
import numpy as np
import polars as pl
import pylab as plt
from scipy.integrate import solve_ivp

from sting import datasets, main
from sting.generator import SynchronousGenerator23A

from sting.generator.core import PowerFlowVariables
from sting.utils.dynamical_systems import make_smooth_step
from sting.utils.transformations import dq02abc, abc2dq0

matplotlib.use('TkAgg')

gen = SynchronousGenerator23A(
    bus="bus_1",
    base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # Shaft, governor and turbine parameters Kundur (page 598)
    h_s=10, kd_w_pu=1,                         # Shaft
    kr_pu=0.05, tau_g_s=0.2,                 # Governor
    tau_rh_s=7.0, f_hp_pu=0.3, tau_ch_s=0.3, # Turbine parameters
    # Machine parameters Kundur (page 155)
    x_d_pu=1.81, x_q_pu=1.76, x_l_pu = 0.15, r_a_pu=0.003, 
    x_td_pu=0.3, x_tq_pu=0.65, x_std_pu=0.23, x_stq_pu=0.25,
    t_td0_s=8.0, t_tq0_s=1, t_std0_s=0.03, t_stq0_s=0.07,
    x_0_pu=0.25,
    # Excitor parameters Kundur (page 364)
    ka_pu=187, ta_s=0.89, te_s=1.15, kf_pu=0.058,
    tf_s=0.62, tb_s=0.06, tc_s=0.173, tau_v_s=0.05,
    ke_pu=1, # TODO: Compute this value
    # Shunt
    csh_pu=0.066, rsh_pu=10,
    # Branch and transformer
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, 
    txr_r1_pu=0.01, txr_x1_pu=0.1, txr_r2_pu=0.02, txr_x2_pu=0.1, 
)

# Power flow
pf_sol = PowerFlowVariables(
    vmag_bus = 1.0,
    vphase_bus= 0,
    p_bus = 1,
    q_bus = 0.2)




gen.power_flow_variables = pf_sol
gen._calculate_emt_initial_conditions()
inputs = {
    "p_set": lambda t: gen.shaft.emt_init.p_ref,
    "v_set": lambda t: gen.exciter.emt_init.v_ref,
    "v_bus_a": lambda t: np.sqrt(2) * pf_sol.vmag_bus * np.cos(pf_sol.vphase_bus * np.pi / 180 + 2 * np.pi * 60 * t),
    "v_bus_b": lambda t: np.sqrt(2) * pf_sol.vmag_bus * np.cos(pf_sol.vphase_bus * np.pi / 180 - (2 * np.pi / 3) + 2 * np.pi * 60 * t),
    "v_bus_c": lambda t: np.sqrt(2) * pf_sol.vmag_bus * np.cos(pf_sol.vphase_bus * np.pi / 180 + (2 * np.pi / 3) + 2 * np.pi * 60 * t),
}

u0 = [u(0) for u in inputs.values()]

gen.define_variables_emt()

dx = gen.get_derivative_state_emt(x=gen.variables_emt.x.init, u=u0)

for i, x in enumerate(np.array(dx) - np.array(gen.variables_emt.x.init)):
    print(gen.variables_emt.x.name[i], x)

print("ok")