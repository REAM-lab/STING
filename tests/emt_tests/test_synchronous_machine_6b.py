import os

import matplotlib
import numpy as np
import polars as pl
import pylab as plt
from scipy.integrate import solve_ivp

from sting import datasets, main
from sting.components.synchronous_machine_6b import SynchronousMachine6B
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
l_d_pu = Ld / l_s_base
l_q_pu = Lq / l_s_base
l_0_pu = L0 / l_s_base

# Field circuit base values
v_fd_base = 1e6
i_fd_base = s_base/v_fd_base
z_fd_base = v_fd_base/i_fd_base
l_fd_base = s_base/(w_base*i_fd_base**2)
l_ad_pu = MF / l_s_base * i_fd_base / i_s_base
l_ffd_pu = LF / l_fd_base
r_fd_pu = rf / z_fd_base


# 1d base values
i_1d_base = MF * i_fd_base / MD
l_1d_base = s_base/(w_base*i_1d_base**2)
z_1d_base = w_base*l_1d_base
l_11d_pu = LD / l_1d_base
l_f1d_pu = MR / l_fd_base * i_1d_base / i_fd_base
l_1df_pu = MR / l_1d_base * i_fd_base / i_1d_base
r_1d_pu = rd / z_1d_base

# Leakage
l_l_pu = l_d_pu - l_ad_pu

# 1q base values
l_aq_pu = l_q_pu - l_l_pu
i_1q_base = l_aq_pu * l_s_base * i_s_base / MQ
l_1q_base = s_base/(w_base*i_1q_base**2)
l_11q_pu = LQ / l_1q_base
l_aq_pu2 = MQ / l_s_base * i_1q_base / i_s_base
z_1q_base = w_base*l_1q_base
r_1q_pu = rq / z_1q_base

print("l_d_pu: ", l_d_pu)
print("l_q_pu: ", l_q_pu)
print("l_0_pu: ", l_0_pu)
print("l_ad_pu: ", l_ad_pu)
print("l_ffd_pu: ", l_ffd_pu)
print("l_11d_pu: ", l_11d_pu)
print("l_f1d_pu: ", l_f1d_pu)
print("l_1df_pu: ", l_1df_pu)
print("l_l_pu: ", l_l_pu)
print("l_aq_pu: ", l_aq_pu)
print("l_11q_pu: ", l_11q_pu)
print("Check: l_aq_pu2: ", l_aq_pu2)



sm = SynchronousMachine6B(
    x_d_pu = l_d_pu,
    x_q_pu = l_q_pu,
    x_0_pu = l_0_pu,
    x_ad_pu = l_ad_pu,
    x_aq_pu = l_aq_pu,
    x_ffd_pu = l_ffd_pu,
    x_f1d_pu = l_f1d_pu,
    x_11d_pu = l_11d_pu,
    x_11q_pu = l_11q_pu,
    r_a_pu = r_a_pu,
    r_fd_pu = r_fd_pu,
    r_1d_pu = r_1d_pu,
    r_1q_pu = r_1q_pu,
    w_base = w_base,
    k1=1,
    k2=1
)


import control as ct
sys = ct.ss(sm.A, sm.B, np.diag([-1,-1,-1,1,1,1]), np.zeros((6,4)))

# Create a sigma plot
"""omg = [1e-2, 1e5]
ct.singular_values_plot(sys, omega_limits=omg)
plt.show()"""

# check if all entries of matrix L are positive
L = sm.L
for i in range(L.shape[0]):
    for j in range(L.shape[1]):
        if L[i,j] < 0:
            print(f"Entry L[{i},{j}] = {L[i,j]} is negative. All entries of matrix L should be positive.")

# Eigenvalues of A
eigenvalues = np.linalg.eigvals(sm.A + sm.N)
for eig in eigenvalues:
    print(f"Eigenvalue: {eig:.4f}, Real: {np.real(eig):.4f}, Imag: {np.imag(eig):.4f}")


# ---------------------------------------------------------------
# Solve the EMT dynamics of a three-phase short circuit in the 
# armature terminals
# ---------------------------------------------------------------

v_fd = 400/v_fd_base
def step(t, x):
    i_d, i_q, i_0, i_fd, i_1d, i_1q, angle = x
    dx  = sm.get_derivatives_step_emt_dq0( i_d, i_q, i_0, i_fd, i_1d, i_1q, 0, 0, 0, v_fd, 1)
    d_angle = w_base 
    return np.concatenate((dx, [d_angle]))

x0 = [0, # i_0
      0, # i_d
      0, # i_q
      v_fd/r_fd_pu, # i_fd
      0, # i_1d
      0, # i_1q 
      np.pi/2] # angle
# Solve
sol = solve_ivp(
    fun=step, 
    y0=x0, 
    t_span=[0, 0.8],
    max_step = 1e-4,
    dense_output=True,
    method="Radau"
    )

#cos(a+pi/2) = -sin(a)
#sin(a+pi/2) = cos(a)

# Transform to abc
i_d = -sol.y[0]
i_q = -sol.y[1]
i_0 = sol.y[2]
i_fd = sol.y[3]
i_1d = sol.y[4]
i_1q = sol.y[5]
angle = sol.y[6]

ia = (i_d * np.cos(angle) - i_q * np.sin(angle))
ib = (i_d * np.cos(angle - 2*np.pi/3) - i_q * np.sin(angle - 2*np.pi/3))
ic = (i_d * np.cos(angle + 2*np.pi/3) - i_q * np.sin(angle + 2*np.pi/3))

# ---------------------------------------------------------------
# Compare results
# ---------------------------------------------------------------

# Validation dataset
df = pl.read_csv(os.path.join("/Users/adamsedlak/Documents/Python/STING/tests/emt_tests/validation_data/", "ch8ex2.csv"))

# Take id column convert to numpy
t_h = df["t"].to_numpy()
theta_h = w_base * t_h + np.pi/2
id_h = df["id"].to_numpy()
iq_h = df["iq"].to_numpy()
ia_h = np.sqrt(2/3) * (id_h * np.cos(theta_h) + iq_h * np.sin(theta_h))
ib_h = np.sqrt(2/3) * (id_h * np.cos(theta_h - 2*np.pi/3) + iq_h * np.sin(theta_h - 2*np.pi/3))
ic_h = np.sqrt(2/3) * (id_h * np.cos(theta_h + 2*np.pi/3) + iq_h * np.sin(theta_h + 2*np.pi/3))


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=False)

ax1.plot(t_h, ia_h)
ax2.plot(t_h, ib_h)
ax3.plot(t_h, ic_h)
ax4.plot(t_h, df["iF"])
ls = "--"
ax1.plot(sol.t, i_s_base*ia, ls=ls)
ax2.plot(sol.t, i_s_base*ib, ls=ls)
ax3.plot(sol.t, i_s_base*ic, ls=ls)
ax4.plot(sol.t, i_fd_base*i_fd, ls=ls)
ax1.set_title(r"$i_a$")
ax2.set_title(r"$i_b$")
ax3.set_title(r"$i_c$")

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlim(0, 0.8)

plt.show()