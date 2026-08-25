import os

import matplotlib
import numpy as np
import polars as pl
import pylab as plt
from scipy.integrate import solve_ivp

from sting import datasets, main
from sting.components import SynchronousMachine6A
from sting.generator.core import PowerFlowVariables
from sting.utils.dynamical_systems import make_smooth_step
from sting.utils.transformations import dq02abc, abc2dq0

matplotlib.use('TkAgg')


import h5py

# Parameters from book
Ld = 0.0072 # H
Lq = 0.0070 # H
LF = 2.5 # H
LD = 0.0068 # H
LQ = 0.0016 # H
MF = 0.100 # H
MD = 0.0054 # H
MQ = 0.0026 # H
MR = 0.125 # H
r = 0.0020 # Ohm
rf = 0.4000 # Ohm
rd = 0.015 # Ohm
rq = 0.015 # Ohm
L0 = 0.0010 # H

# Stator base values
s_base = 500e6 # VA
v_rms_line_base = 30e3 # V
f_base = 60 # Hz
w_base = 2*np.pi*f_base # rad/s
v_rms_base = v_rms_line_base/np.sqrt(3)
v_s_base = v_rms_base*np.sqrt(2)
i_s_base = 2 * s_base/(3*v_s_base)
z_s_base = v_s_base/i_s_base
l_s_base = z_s_base/w_base
r_a_pu = r / z_s_base

# Field base values
l_f_base = 2/3 * l_s_base * (MR**2)/(MD**2)
i_f_base = np.sqrt(s_base/(l_f_base * w_base))
z_f_base = l_f_base * w_base
r_fd_pu = rf / z_f_base
s_f_base = w_base * l_f_base * i_f_base**2

# 1d base values
l_1d_base = MD**2 / MF**2 * l_f_base
i_1d_base = np.sqrt(s_base/(l_1d_base * w_base))
z_1d_base = l_1d_base * w_base
r_1d_pu = rd / z_1d_base
s_1d_base = w_base * l_1d_base * i_1d_base**2

# Compute l_ad_pu
l_ad_pu = MF / l_s_base * i_f_base / i_s_base

# Compute l_ffd_pu
l_ffd_pu = LF / l_f_base 
l_f1d_pu = MR / l_f_base * i_1d_base / i_f_base
l_fd_pu = l_ffd_pu - l_f1d_pu

# Compute l_11d_pu
l_11d_pu = LD / l_1d_base
l_1df_pu = MR / l_1d_base * i_f_base / i_1d_base
l_1d_pu = l_11d_pu - l_1df_pu

# Compute l_d_pu, l_q_pu
l_d_pu = Ld / l_s_base
l_q_pu = Lq / l_s_base
l_0_pu = L0 / l_s_base

# Compute l_l_pu
l_l_pu = l_d_pu - l_ad_pu

# Compute l_aq_pu
l_aq_pu = l_q_pu - l_l_pu

# Compute 1q base values
i_1q_base = l_aq_pu * l_s_base * i_s_base / MQ
l_1q_base = s_base / (i_1q_base**2 * w_base) 
z_1q_base = l_1q_base * w_base
r_1q_pu = rq / z_1q_base
s_1q_base = w_base * l_1q_base * i_1q_base**2

# Compuate l_11q_pu
l_11q_pu = LQ / l_1q_base
l_1q_pu = l_11q_pu - l_aq_pu

l_ad_pu_2 = MD / l_s_base * i_1d_base / i_s_base
l_ad_pu_3 = MR / l_f_base * i_1d_base / i_f_base

# Check if s_base in all rotor circuits are equal
if not np.isclose(s_f_base, s_1d_base) or not np.isclose(s_f_base, s_1q_base) or not np.isclose(s_f_base, s_base):
    raise ValueError("Base power in all rotor circuits are not equal. Check the parameters.")



# Parameters according to Kundur's book, "Power System Stability and Control", 1994, page 155
sm = SynchronousMachine6A(
    # Paramters
    x_0_pu = l_0_pu,
    x_d_pu = l_d_pu,
    x_q_pu = l_q_pu,
    x_ad_pu = l_ad_pu,
    x_aq_pu = l_aq_pu,
    x_fd_pu = l_fd_pu,
    x_1d_pu = l_1d_pu,
    x_1q_pu = l_1q_pu,
    r_a_pu = r_a_pu,
    r_fd_pu = r_fd_pu,
    r_1d_pu = r_1d_pu,
    r_1q_pu = r_1q_pu,
    w_base = 2*np.pi*f_base
)

print("Computation of Kundur's standard parameters:")
print(f'x_ad: {sm.x_ad_pu}, x_aq: {sm.x_aq_pu}, x_fd: {sm.x_fd_pu}, x_1d: {sm.x_1d_pu}, x_1q: {sm.x_1q_pu}')
print(f'r_fd: {sm.r_fd_pu}, r_1d: {sm.r_1d_pu}, r_1q: {sm.r_1q_pu}')

# check if all entries of matrix L are positive
L = sm.L
for i in range(L.shape[0]):
    for j in range(L.shape[1]):
        if L[i,j] < 0:
            raise ValueError(f"Entry L[{i},{j}] = {L[i,j]} is negative. All entries of matrix L should be positive.")

# Eigenvalues of A
eigenvalues = np.linalg.eigvals(sm.A + sm.N)
for eig in eigenvalues:
    print(f"Eigenvalue: {eig:.4f}, Real: {np.real(eig):.4f}, Imag: {np.imag(eig):.4f}")