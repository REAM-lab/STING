"""
This is a skeleton script for debugging interconnection matrices for the component connection method.
1. Paste your interconnection matrices
2. Define your symbolic inputs and outputs
3. Check that each row of the resulting vector matches it's expected variable


Recall the transformation from DQ to dq  
    i_d =  i_D*cos + i_Q*sin
    i_q = -i_D*sin + i_Q*cos

Active and reactive power
    p = v_d * i_d + v_q * i_q
    q = v_q * i_d - v_d * i_q
"""

import numpy as np
import sympy as sp

from sympy import Matrix
from sympy.physics.quantum import TensorProduct

from sting import datasets, main
from sting.generator import SynchronousGenerator23A

# 1. Replace this device with a class instance that you want to test
device =  SynchronousGenerator23A(
    bus="bus_2", name="gen1",
    # Power flow 
    minimum_active_power_MW=-100, maximum_active_power_MW=-50, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=10,
    # Per unit system
    base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # Shaft, governor and turbine parameters Kundur (page 598)
    h_s=2.0, kd_w_pu=1,                      # Shaft
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
    ke_pu=1,
    # Shunt
    csh_pu=0.066, rsh_pu=10,
    # Branch and transformer
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, 
    txr_r1_pu=0.01, txr_x1_pu=0.1, txr_r2_pu=0.02, txr_x2_pu=0.1, 
)

L11, L12, L21, L22 = device.get_interconnections_ssm(1,1,1,1,0)

# --------------
# Inputs/Outputs
# --------------
def vectorize(x):
    symbol_creator = np.vectorize(sp.Symbol)
    return symbol_creator(x)

# Replace these each vector with a list of strings containing your 
# state, output, and input names. For instance: ['i_bus_d', 'i_bus_q', ...] 


u_stack = vectorize(['p_ref', 'i_d', 'i_q', 'v_d', 'v_q', 'p_ref', 'w', 'u_vlv', 'v_d',
       'v_q', 'v_0', 'v_fd', 'w', 'v_d', 'v_q', 'v_ref', 'v_mag',
       'v_stab', 'i_sh_D', 'i_sh_Q', 'v_from_D', 'v_from_Q', 'v_to_D',
       'v_to_Q'])
y_stack = vectorize(['angle', 'w', 'x_gov', 'y0', 'i_d', 'i_q', 'i_0', 'i_fd', 'i_1d',
       'i_1q', 'i_2q', 'v_mag', 'v_fd', 'v_sh_D', 'v_sh_Q', 'i_br_D',
       'i_br_Q'])
u_grid = vectorize(["p_ref", "v_ref", "v_bus_D", "v_bus_Q"])
y_grid = vectorize(["i_bus_D", "i_bus_Q"])


# ---------------------------
# Component Connection Method
# ---------------------------
u_out = Matrix(L11) @ y_stack + Matrix(L12) @ u_grid
y_out = Matrix(L21) @ y_stack + Matrix(L22) @ u_grid

print("Expected vs. actual inputs")
for u_expected, u_actual in zip(u_stack, u_out):
    print(u_expected, " == ",sp.nsimplify(u_actual))

print("\nExpected vs. actual outputs")
for y_expected, y_actual in zip(y_grid, y_out):
    print(y_expected, " == ",sp.nsimplify(y_actual))

print("ok")