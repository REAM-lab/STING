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
from sting.generator import GFLI16A

# 1. Replace this device with a class instance that you want to test
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


x = np.array([[3],[2]])

L11, L12, L21, L22, M1, M2 = gfli_1.get_interconnections_qbm(x,x,x,x)

# --------------
# Inputs/Outputs
# --------------
def vectorize(x):
    symbol_creator = np.vectorize(sp.Symbol)
    return symbol_creator(x)

# Replace these each vector with a list of strings containing your 
# state, output, and input names. For instance: ['i_bus_d', 'i_bus_q', ...] 


u_grid = vectorize(["p_ref", "q_ref","w_set","ω_slack","one", "v_bus_D", "v_bus_Q"])
y_grid = vectorize(["i_bus_D", "i_bus_Q"])

y_stack = vectorize(['w', 'sin', 'cos', 'i_ref_d', 'i_ref_q', 'v_out_d', 'v_out_q',
       'i_vsc_d', 'i_vsc_q', 'i_bus_d', 'i_bus_q', 'v_sh_d', 'v_sh_q']) 
x_stack = vectorize(['v_pll_q', 'z_pll', 'sin', 'cos', 'z_apc', 'z_rpc', 'z_cc_d',
       'z_cc_q', 'i_vsc_d', 'i_vsc_q', 'i_bus_d', 'i_bus_q', 'v_sh_d',
       'v_sh_q'])
u_stack = vectorize(['w_set', 'w_slack', 'one', 'v_bus_D', 'v_bus_Q', 'p_ref', 'p_apc',
       'q_ref', 'q_rpc', 'i_d_ref', 'i_q_ref', 'i_d', 'i_q', 'v_d', 'v_q',
       'w', 'w_slack', 'v_from_d', 'v_from_q', 'v_to_d', 'v_to_q',
       'w_slack', 'v_from_d', 'v_from_q', 'v_to_d', 'v_to_q', 'w_slack',
       'i_sh_d', 'i_sh_q'])

# ---------------------------
# Component Connection Method
# ---------------------------
u_out = Matrix(L11) @ y_stack + Matrix(L12) @ u_grid  + Matrix(M1)@ TensorProduct(x_stack, x_stack) + Matrix(M2) @ TensorProduct(u_grid, x_stack)
y_out = Matrix(L21) @ y_stack + Matrix(L22) @ u_grid

print("Expected vs. actual inputs")
for u_expected, u_actual in zip(u_stack, u_out):
    print(u_expected, " == ",sp.nsimplify(u_actual))

print("\nExpected vs. actual outputs")
for y_expected, y_actual in zip(y_grid, y_out):
    print(y_expected, " == ",sp.nsimplify(y_actual))

print("ok")