

import os

import matplotlib
import numpy as np
import polars as pl
import pylab as plt
from scipy.integrate import solve_ivp

from sting import datasets, main
from sting.generator import SM8A
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.dynamical_systems import DynamicalVariables
from sting.utils.transformations import dq02abc

matplotlib.use('TkAgg')

"""

% Combine into a MATLAB table and name the columns
T = table(t, i(:,1), i(:,2), i(:,3), i(:,4), i(:,5), 'VariableNames', {'t', 'id', 'iF', 'iD', 'iq', 'iQ'});

% Write table to CSV (headers are included automatically)
writetable(T, 'ch8ex2.csv');

"""

# ---------------------------------------------------------------
# Parameters of a 500 MVA, 30 kV Synchronous Machine. The DC field 
# voltage is VF = 400 (Hadi Saadat, Ch8 Ex2).
# ---------------------------------------------------------------
LF = 2.500
LD = 0.0068
LQ = 0.0016
Ld = 0.0072
Lq = 0.0070
L0 = 0.0010

MF = 0.100
MD = 0.0054
MQ = 0.0026
MR = 0.1250

ra = 0.002
rF = 0.4000
rD = 0.015
rQ = 0.0150

rated_MVA = 500
rated_kV = 30
rated_frequency_Hz = 60
excitation_voltage_kV = 0.400
# ---------------------------------------------------------------
# Converting Saadat's notation to Kundur's notation
# ---------------------------------------------------------------
# Total self inductances
l_ffd = LF # Field d-axis
l_kkd = LD # Damper d-axis
l_kkq = LQ # Damper q-axis
l_d = Ld   # Stator d-axis
l_q = Lq   # Stator q-axis
l_0 = L0   # Stator 0-axis
# Mutual inductances
l_afd = MF # Armature-field
l_akd = MD # Armature-damper
l_akq = MQ # Armature-damper
l_fkd = MR # Field-damper 
# Resistances
r_a = ra
r_fd = rF
r_kd = rD
r_kq = rQ

# ---------------------------------------------------------------
# Base quantities for the machine
# ---------------------------------------------------------------
# Angular velocity in electrical rad/s
w_base = 2 * np.pi * rated_frequency_Hz

# Stator
# ----------------------
# Peak phase-to-neutral rated voltage 
e_s_base_kV = (2/3)**0.5 * rated_kV 
# Peak line current (Kundur, 85)
i_s_base_kA = rated_MVA / (1.5 * e_s_base_kV )
z_s_base_ohms = e_s_base_kV / i_s_base_kA 
l_s_base_H = z_s_base_ohms / w_base

# Rotor
# ----------------------
# We need to establish a base voltage or base current for
# the rotor circuits (Kundur, pg 81). Without a leakage 
# inductance we will establish a base current for the field 
# such that i_fd is near 1 pu.  
#
# The field voltage is 400V and r_f = 0.4 ohms, thus i_fd 
# at t = 0 will be 400V/0.4ohms = 1000V or 1kV. Recovering
# l_ad from the assumed base field current as
#       i_fd_base_kA = (l_ad / l_afd) * i_s_base_kA     (3.113)
#       l_afd * (i_fd_base_kA / i_s_base_kA) = l_ad
# Then we can recover the leakage inductance and l_aq
#       l_ad = l_d - l_l        (3.111)
#       l_aq = l_q - l_l        (3.112)
i_fd_base_kA = excitation_voltage_kV / r_fd # We pick i_fd_base
e_fd_base_kV = rated_MVA / i_fd_base_kA     # ...and v_fd base follows
# Recover leakage inductance
l_ad = l_afd * (i_fd_base_kA / i_s_base_kA)
l_l = l_d - l_ad
l_aq = l_q - l_l

# See Kundur 85-86
z_fd_base_ohms = rated_MVA / i_fd_base_kA**2
l_fd_base_H = z_fd_base_ohms / w_base
# Damping d-axis
i_kd_base_kA = (l_ad / l_akd) * i_s_base_kA
z_kd_base_ohms = rated_MVA / i_kd_base_kA**2
l_kd_base_H = z_kd_base_ohms / w_base
# Damping q-axis
i_kq_base_kA = (l_aq / l_akq) * i_s_base_kA
z_kq_base_ohms = rated_MVA / i_kq_base_kA**2
l_kq_base_H = z_kq_base_ohms / w_base

# ---------------------------------------------------------------
# Per unit parameters
# ---------------------------------------------------------------

# See Kundur page 76-77
r_a_pu = r_a / z_s_base_ohms
r_fd_pu = r_fd / z_fd_base_ohms
r_1d_pu = r_kd / z_kd_base_ohms
r_1q_pu = r_kq / z_kq_base_ohms

# In the per unit system chosen by Kundur the following hold (pg. 86)
#   L_afd = L_fad = L_akd = L_kda = L_ad
#   L_akq = L_kqa = L_aq
#   L_fkd = L_kdf
x_l_pu = l_l / l_s_base_H

x_ad_pu = 1.5 * (l_afd / l_fd_base_H) * (i_s_base_kA / i_fd_base_kA) # (3.101)
x_f1d_pu = (l_fkd/l_fd_base_H) * (i_kd_base_kA / i_fd_base_kA)       # (3.102)
x_aq_pu = 1.5 * (l_akq / l_kq_base_H) * (i_s_base_kA / i_kq_base_kA) # (3.105)

# Total inductances
x_0_pu = l_0 / l_s_base_H
x_d_pu = x_ad_pu + x_l_pu  # (3.111)
x_q_pu =  x_aq_pu + x_l_pu # (3.112)

x_fd_pu = (l_ffd / l_fd_base_H) - x_f1d_pu # (3.135)
x_1d_pu = (l_kkd / l_kd_base_H) - x_f1d_pu # (3.136)
x_1q_pu = (l_kkq / l_kq_base_H) - x_aq_pu  # (3.137)

# Saadat's model has only one damper in the q-axis. We will assume 
# parameters for the second q-axis using those from Kundur, Example 4.1
# page 153.
r_2q_pu = 1e5#0.006194377297124655
x_2q_pu = 1e5#0.7252252252252251

"""
x_ad: 1.6600000000000001, x_aq: 1.61, x_fd: 0.16490066225165562, x_1d: 0.1714285714285715, x_1q: 0.7252252252252251, x_2q: 0.125
r_fd: 0.0006050874188521342, r_1d: 0.02842052555212418, r_1q: 0.006194377297124655, r_2q: 0.023683771293436805
"""
# ---------------------------------------------------------------
# Converting to standard parameters 
# ---------------------------------------------------------------

# d-axis transient parameters (Kundur, pg 150)
t_1 = (x_ad_pu + x_fd_pu) / r_fd_pu
t_2 = (x_ad_pu + x_1d_pu) / r_1d_pu
t_3 = (1 / r_1d_pu) * (x_1d_pu + (x_ad_pu*x_fd_pu) / (x_ad_pu + x_fd_pu))
t_4 = (1 / r_fd_pu) * (x_fd_pu + (x_ad_pu*x_l_pu) / (x_ad_pu + x_l_pu) )
t_5 = (1 / r_1d_pu) * (x_1d_pu + (x_ad_pu*x_l_pu) / (x_ad_pu + x_l_pu))

num = x_ad_pu*x_l_pu*x_fd_pu
den = (x_ad_pu*x_l_pu + x_ad_pu*x_fd_pu + x_fd_pu*x_l_pu)
t_6 = (1 / r_1d_pu) * (x_1d_pu + num / den)

x_td_pu = x_d_pu * (t_4 / t_1)
x_std_pu = x_d_pu * (t_4*t_6) / (t_1*t_3)
t_td0_s = t_1 / w_base
t_std0_s= t_3 / w_base

# q-axis transient parameters (Kundur, pg 147)
t_tq0_s = (x_aq_pu + x_1q_pu) / (r_1q_pu*w_base)
t_stq0_s = (1 / (w_base*r_2q_pu)) * (x_2q_pu + (x_aq_pu*x_1q_pu) / (x_aq_pu + x_1q_pu))

num = x_aq_pu*x_1q_pu*x_2q_pu
den = x_aq_pu*x_1q_pu + x_aq_pu*x_2q_pu + x_1q_pu*x_2q_pu
x_stq_pu = x_l_pu + num / den
x_tq_pu = x_l_pu + (x_aq_pu*x_1q_pu) / (x_aq_pu + x_1q_pu)



# ---------------------------------------------------------------
# Construct the synchronous machine model
# ---------------------------------------------------------------
sm = SM8A(
    name="SM8A", bus="bus_2",
    # Power flow 
    minimum_active_power_MW=None, maximum_active_power_MW=None, minimum_reactive_power_MVAR=None, maximum_reactive_power_MVAR=None,
    cost_variable_USDperMWh=None, base_power_MVA=rated_MVA, base_voltage_kV=rated_kV, base_frequency_Hz=rated_frequency_Hz,
    # Paramters
    x_d_pu=x_d_pu, x_q_pu=x_q_pu, x_0_pu=x_0_pu, 
    x_l_pu=x_l_pu, x_f1d_pu=x_f1d_pu, r_a_pu=r_a_pu,
    x_td_pu=x_td_pu, x_tq_pu=x_tq_pu, x_std_pu=x_std_pu, x_stq_pu=x_stq_pu,
    t_td0_s=t_td0_s, t_tq0_s=t_tq0_s, t_std0_s=t_std0_s, t_stq0_s=t_stq0_s,
    k1 = 1, k2 = 1,
)

# Confirm that standard parameters we computed correctly
for var in ["x_ad_pu", "x_aq_pu", "x_fd_pu", "x_1d_pu", "x_1q_pu", "x_2q_pu", "r_fd_pu", "r_1d_pu", "r_1q_pu", "r_2q_pu"]:
    assert np.isclose(getattr(sm, var), globals()[var])


# ---------------------------------------------------------------
# Solve the EMT dynamics of a three-phase short circuit in the 
# armature terminals
# ---------------------------------------------------------------

def step(t, x):
    sm.variables_emt = VariablesEMT(
        u=DynamicalVariables(value=[excitation_voltage_kV/e_fd_base_kV, 0, 0, 0], name=["v_fd", 'v_a', "v_b", "v_c"]),
        x=DynamicalVariables(value=x, name=["angle", "i_d", "i_q", "i_0", "i_fd", "i_1d", "i_1q", "i_2q"]),
        y=None
    )
    dx  = sm.get_derivative_state_emt(x, u=[excitation_voltage_kV/e_fd_base_kV, 0, 0, 0])
    return dx

# Solve
sol = solve_ivp(
    fun=step, 
    y0=[0, 0, 0, 0, 1, 0, 0, 0], 
    t_span=[0, 0.8],
    max_step = 1e-4,
    dense_output=True,
    method="Radau"
    )

# ---------------------------------------------------------------
# Compare results
# ---------------------------------------------------------------

# Validation dataset
df = pl.read_csv(os.path.join(os.getcwd(), "tests", "emt_tests", "validation_data", "ch8ex2.csv"))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(df["t"], df["id"])
ax2.plot(df["t"], df["iq"])
ax3.plot(df["t"], df["iF"])

ls = "--"
ax1.plot(sol.t, -1e3*i_s_base_kA*sol.y[1], ls=ls)
ax2.plot(sol.t, 1e3*i_s_base_kA*sol.y[2], ls=ls)
ax3.plot(sol.t, 1e3*i_fd_base_kA*sol.y[4], ls=ls)

ax1.set_title(r"$i_d$")
ax2.set_title(r"$i_q$")
ax3.set_title(r"$i_{fd}$")

for ax in (ax1, ax2, ax3):
    ax.set_xlim(0, 0.8)

# Plot results
titles = [r"angle", r"$i_d$", r"$i_q$", r"$i_0$", r"$i_{fd}$", r"$i_{1d}$", r"$i_{1q}$", r"$i_{2q}$"]

plt.show()

print("ok")