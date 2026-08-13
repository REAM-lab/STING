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
from sting.generator import GFMI18B

# 1. Replace this device with a class instance that you want to test
device = GFMI18B(
    name="santiago_gfmi", bus="santiago",
    # Power flow 
    minimum_active_power_MW=80, maximum_active_power_MW=80, minimum_reactive_power_MVAR=50, maximum_reactive_power_MVAR=51,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.005, xf1_pu=0.15, csh_pu=0.066, rsh_pu=1,
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

L11, L12, L21, L22, M1, M2 = device.get_interconnections_qbm()

# 2. Replace these components with the assumed order of your device internal components
components = [
    device.virtual_inertia, 
    device.voltage_droop, 
    device.voltage_controller, 
    device.current_controller, 
    device.lcl_br1, 
    device.lcl_br2, 
    device.lcl_sh]


# --------------
# Inputs/Outputs
# --------------
def vectorize(x):
    symbol_creator = np.vectorize(sp.Symbol)
    return symbol_creator(x)

# Replace these each vector with a list of strings containing your 
# state, output, and input names. For instance: ['i_bus_d', 'i_bus_q', ...] 

x_stack = vectorize(['w', 'sin', 'cos', 'q_f', 'z_vc_d', 'z_vc_q', 'z_cc_d', 'z_cc_q',
       'i_br_d', 'i_br_q', 'i_br_D', 'i_br_Q', 'v_sh_D', 'v_sh_Q'])
u_stack = vectorize(['p_ref', 'one', 'p', 'q_ref', 'v_ref', 'q', 'v_ref_d', 'v_ref_q',
       'v_d', 'v_q', 'i_d', 'i_q', 'i_d_ref', 'i_q_ref', 'i_d', 'i_q',
       'v_d', 'v_q', 'v_from_d', 'v_from_q', 'v_to_d', 'v_to_q', 'w',
       'v_from_D', 'v_from_Q', 'v_to_D', 'v_to_Q', 'i_sh_D', 'i_sh_Q'])
y_stack = vectorize(['w', 'sin', 'cos', 'v_d_ref', 'v_q_ref', 'i_out_d', 'i_out_q',
       'v_out_d', 'v_out_q', 'i_br_d', 'i_br_q', 'i_br_D', 'i_br_Q',
       'v_sh_D', 'v_sh_Q'])
u_grid = vectorize(["p_ref",  "q_ref",  "v_ref",  "one",   "v_bus_D", "v_bus_Q"])
y_grid = vectorize(["i_bus_D", "i_bus_Q"])


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