import os

import matplotlib
import numpy as np
import polars as pl
import pylab as plt
from scipy.integrate import solve_ivp

from sting import datasets, main
from sting.components import SynchronousMachine7A
from sting.generator.core import PowerFlowVariables
from sting.utils.dynamical_systems import make_smooth_step
from sting.utils.transformations import dq02abc, abc2dq0

matplotlib.use('TkAgg')


import h5py



# -------------------------------------------------------
# Load components and a power flow solution
# -------------------------------------------------------

# Parameters according to Kundur's book, "Power System Stability and Control", 1994, page 155
sm = SynchronousMachine7A(
    w_base=2*np.pi*60,
    # Paramters
    x_d_pu=1.81, x_q_pu=1.76, x_l_pu = 0.15, r_a_pu=0.003, 
    x_td_pu=0.3, x_tq_pu=0.65, x_std_pu=0.23, x_stq_pu=0.25,
    t_td0_s=8.0, t_tq0_s=1, t_std0_s=0.03, t_stq0_s=0.07,
    x_0_pu=0.25
)

print("Computation of Kundur's standard parameters:")
print(f'x_ad: {sm.x_ad_pu}, x_aq: {sm.x_aq_pu}, x_fd: {sm.x_fd_pu}, x_1d: {sm.x_1d_pu}, x_1q: {sm.x_1q_pu}, x_2q: {sm.x_2q_pu}')
print(f'r_fd: {sm.r_fd_pu}, r_1d: {sm.r_1d_pu}, r_1q: {sm.r_1q_pu}, r_2q: {sm.r_2q_pu}')

# Power flow
v_bus_mag = 1.0
v_bus_angle = 0
p_bus = 1
q_bus = 0.2

v_DQ = v_bus_mag * np.exp(v_bus_angle * np.pi / 180 * 1j)

# -------------------------------------------------------
# Compute initial conditions for EMT simulation
# -------------------------------------------------------
sm.get_steady_state(v_ref_mag=v_bus_mag, v_ref_angle=v_bus_angle, p_ref=p_bus, q_ref=q_bus)
y0 = np.array([sm.emt_init.angle, sm.emt_init.i_d, sm.emt_init.i_q, sm.emt_init.i_0, sm.emt_init.i_fd, sm.emt_init.i_1d, sm.emt_init.i_1q, sm.emt_init.i_2q])

print("Initial field circuit voltage: ", sm.emt_init.v_fd)
print("Initial angle: ", sm.emt_init.angle * 180 / np.pi)
print("Initial current magnitude (rms): ", np.sqrt(sm.emt_init.i_d**2 + sm.emt_init.i_q**2))
print("Initial current angle (in the grid's frame): ", np.arctan2(sm.emt_init.i_Q, sm.emt_init.i_D) * 180 / np.pi)


# [0 -37.60162713479158 1.019803902718557 1.019803902718557 1.019803902718557 -11.3099 -131.31 108.69 2.266416388919054]



# -------------------------------------------------------
# Simulation inputs
# -------------------------------------------------------

def v_step(t):

    if t > 1:
        delta = np.sin(20*t)
    else:
        delta = 0

    return sm.emt_init.v_fd + delta*0.1


def w_step(t):
    if t > 1:
        delta = np.cos(30*t)
    else:
        delta = 0

    return 1 + delta*0.1


inputs = {
    "v_bus_a": lambda t: np.sqrt(2) * v_bus_mag * np.cos(v_bus_angle * np.pi / 180 + 2 * np.pi * 60 * t),
    "v_bus_b": lambda t: np.sqrt(2) * v_bus_mag * np.cos(v_bus_angle * np.pi / 180 - (2 * np.pi / 3) + 2 * np.pi * 60 * t),
    "v_bus_c": lambda t: np.sqrt(2) * v_bus_mag * np.cos(v_bus_angle * np.pi / 180 + (2 * np.pi / 3) + 2 * np.pi * 60 * t),
    "v_fd": v_step,
    "w": w_step
}

# -------------------------------------------------------
# Solve for EMT dynamics
# -------------------------------------------------------

def wrap(func):
    def step(t, x):
        angle, \
        i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q = x
    
        # Get inputs
        v_fd, v_bus_a, v_bus_b, v_bus_c = inputs["v_fd"](t), inputs["v_bus_a"](t), inputs["v_bus_b"](t), inputs["v_bus_c"](t)
        w = inputs["w"](t)
        # Transform currents and voltages to dq reference frame
        v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, angle)
    
        # Get derivatives of the state variables
    
        #  i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, v_d, v_q, v_0, v_fd, w
        di_dt = func(i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, v_bus_d, v_bus_q, 0, v_fd, w)
    
        # Angle
        dangle_dt = sm.w_base * w
    
        dx_dt = np.concatenate(([dangle_dt], di_dt))
    
        return dx_dt
    
    return step




# Compute initial conditions and small signal model 
init = sm.emt_init
x0 = np.array([init.i_d, init.i_q, init.i_0, init.i_fd, init.i_1d, init.i_1q, init.i_2q])
u0 = np.array([init.v_d, init.v_q, init.v_0, init.v_fd,  1])

ssm = sm.get_small_signal_model(*x0, *u0)
qbm = sm.get_quadratic_bilinear_model(*x0, *u0)

def sm_step(i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, v_d, v_q, v_0, v_fd, w):
    x = np.array([i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q]) - x0
    u = np.array([v_d, v_q, v_0, v_fd, w]) - u0
    return ssm.A @ x + ssm.B @ u


def qb_step(i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, v_d, v_q, v_0, v_fd, w):
    x = np.array([i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q])
    u = np.array([v_d, v_q, v_0, v_fd, w])
    return qbm.A @ x + qbm.B @ u + qbm.H @ np.kron(x,x) + qbm.N @ np.kron(u, x)


emt_dynamics = wrap(sm.get_derivatives_step_emt_dq0)
qbm_dynamics = wrap(qb_step)
ssm_dynamics = wrap(sm_step)


t_max = 4
# Solve
settings = {
    "t_span": [0,t_max],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}


emt_sol = solve_ivp(emt_dynamics, y0=y0, **settings)
qbm_sol = solve_ivp(qbm_dynamics, y0=y0, **settings)
ssm_sol = solve_ivp(ssm_dynamics, y0=y0, **settings)
ssm_sol.y += y0.reshape(-1, 1)

def sol_to_dataframe(sol):
    # Extract STING solution
    t = np.linspace(0, t_max, 1000)
    angle, i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q = sol.sol(t)
    df = pl.DataFrame(
        {'time': t, 'i_d': i_d, 'i_q': i_q, 'i_fd': i_fd, 'i_1d': i_1d, 'i_1q': i_1q, 'i_2q': i_2q}
    )

    return df

df_emt = sol_to_dataframe(emt_sol)
df_qbm = sol_to_dataframe(qbm_sol)
df_ssm = sol_to_dataframe(ssm_sol)


# -------------------------------------------------------
# Compare solutions
# -------------------------------------------------------

# Load MATLAB solution: You will need to run the associated simulink file.
file_path = os.path.join(os.getcwd(), "tests", "emt_tests", "matlab_data.mat")
with h5py.File(file_path, 'r') as file:
    # Check the variable names
    print(list(file.keys()))
    
    data = file['ans'][:]

# Simulink outputs follow the same schema as STING
df_matlab = pl.DataFrame(data, schema=df_emt.columns)

fig, ax = plt.subplots(3, 2, sharex=True)
axs = ax.flatten()
ls =["-", "-", "--", "-."]
label = ["MATLAB", "EMT", "QBM", "SSM"]

for j, df in enumerate([df_matlab, df_emt, df_qbm, df_ssm]):

    for i, col in enumerate(["i_d", "i_q", "i_fd", "i_1d", "i_1q", "i_2q"]):
        axs[i].set_ylabel(col)
        axs[i].plot(df['time'], df[col], label=label[j], ls=ls[j])

plt.legend()
plt.show()