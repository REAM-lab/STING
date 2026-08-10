"""
This is a skeleton script for debugging interconnection matrices for the component connection method.
1. Paste your interconnection matrices
2. Define your symbolic inputs and outputs
3. Check that each row of the resulting vector matches it's expected variable
"""

import numpy as np
import sympy as sp

from sympy import Matrix
from sympy.physics.quantum import TensorProduct

# ----------------
# Interconnections
# ----------------
# Paste your interconnection matrices here
# where L11 = F, L12 = G, L21 = H, L22 = L 
L11, L12, L21, L22, M1, M2 = None

# --------------
# Inputs/Outputs
# --------------
def vectorize(x):
    symbol_creator = np.vectorize(sp.Symbol)
    return symbol_creator(x)

# Replace these each blank list with a list of strings containing your 
# state, output, and input names. For instance: ['i_bus_d', 'i_bus_q', ...] 
x_stack = vectorize([])
u_stack = vectorize([])
y_stack = vectorize([])
u_grid = vectorize([])
y_grid = vectorize([])


# ---------------------------
# Component Connection Method
# ---------------------------
u_out = Matrix(L11) @ y_stack + Matrix(L12) @ u_grid # + Matrix(M1)@ TensorProduct(x_stack, x_stack) + Matrix(M2) @ TensorProduct(u_grid, x_stack)
y_out = Matrix(L21) @ y_stack + Matrix(L22) @ u_grid

print("Expected vs. actual inputs")
for u_expected, u_actual in zip(u_stack, u_out):
    print(u_expected, " == ",u_actual)

print("\nExpected vs. actual outputs")
for y_expected, y_actual in zip(y_grid, y_out):
    print(y_expected, " == ",y_actual)

print("ok")