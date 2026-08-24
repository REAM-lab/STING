import os

import matplotlib
import numpy as np
import polars as pl
import pylab as plt
import scipy
from scipy.integrate import solve_ivp
import h5py

from sting import datasets, main
from sting.generator import SM8A
from sting.generator.core import PowerFlowVariables
from sting.utils.dynamical_systems import make_smooth_step
from sting.utils.transformations import dq02abc, abc2dq0

matplotlib.use('TkAgg')

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# Parameters according to Kundur's book, "Power System Stability and Control", 1994, page 155
sm = SM8A(
    name="SM8A", bus="bus_2",
    # Power flow 
    minimum_active_power_MW=-100, maximum_active_power_MW=-50, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
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

# Compute initial conditions for EMT simulation
sm._calculate_emt_initial_conditions(v_bus_mag=v_bus_mag, v_bus_angle=v_bus_angle, p_bus=p_bus, q_bus=q_bus)
x0 = np.array([sm.emt_init.angle, sm.emt_init.i_d, sm.emt_init.i_q, sm.emt_init.i_0, sm.emt_init.i_fd, sm.emt_init.i_1d, sm.emt_init.i_1q, sm.emt_init.i_2q])

print("Initial field circuit voltage: ", sm.emt_init.v_fd)

print("Initial angle: ", sm.emt_init.angle * 180 / np.pi)
print("Initial current magnitude (rms): ", np.sqrt(sm.emt_init.i_d**2 + sm.emt_init.i_q**2))
print("Initial current angle (in the grid's frame): ", np.arctan2(sm.emt_init.i_Q, sm.emt_init.i_D) * 180 / np.pi)

# Check initial conditions
#L = sm.L
#R = sm.R
#T = sm.T
#i = np.array([sm.emt_init.i_d, sm.emt_init.i_q, sm.emt_init.i_0, sm.emt_init.i_fd, sm.emt_init.i_1d, sm.emt_init.i_1q, sm.emt_init.i_2q])
#v = np.array([sm.emt_init.v_d, sm.emt_init.v_q, sm.emt_init.v_0, sm.emt_init.v_fd, 0, 0, 0])
#wb = sm.w_base
#di_dt = np.linalg.solve(L, wb * v - wb * 1 * T @ L @ i + wb *R @ i)
#d =  wb * v - wb * 1 * T @ L @ i + wb *R @ i


inputs = {
    "v_bus_a": lambda t: np.sqrt(2) * v_bus_mag * np.cos(v_bus_angle * np.pi / 180 + 2 * np.pi * 60 * t),
    "v_bus_b": lambda t: np.sqrt(2) * v_bus_mag * np.cos(v_bus_angle * np.pi / 180 - 2 * np.pi / 3 + 2 * np.pi * 60 * t),
    "v_bus_c": lambda t: np.sqrt(2) * v_bus_mag * np.cos(v_bus_angle * np.pi / 180 + 2 * np.pi / 3 + 2 * np.pi * 60 * t),
    "v_fd": make_smooth_step(step_time=1, initial_value=sm.emt_init.v_fd, final_value=sm.emt_init.v_fd + 0.5, transient_width=5e-3)
}

# eigenvalues
#M = np.linalg.solve(L, wb * R - wb * 1 * T @ L)
#lamb = np.linalg.eigvals(M)
#large_eigenvalue = np.max(np.real(lamb))
#print(f"Eigenvalues of the linearized system: {lamb}")

def emt_dynamics(t, x):
    """Wrapper function for ODE simulation step"""
    angle, \
    i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q = x

    # Get inputs
    v_fd, v_bus_a, v_bus_b, v_bus_c = inputs["v_fd"](t), inputs["v_bus_a"](t), inputs["v_bus_b"](t), inputs["v_bus_c"](t)

    # Transform currents and voltages to dq reference frame
    v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, angle)

    # Get derivatives of the state variables
    di_dt = sm.get_derivatives_step_emt_dq0(i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q, v_bus_d, v_bus_q, 0, v_fd, 1)

    # Angle
    dangle_dt = sm.w_base

    dx_dt = np.concatenate(([dangle_dt], di_dt))

    return dx_dt

t_max = 4.0
# Solve
settings = {
    "t_span": [0,t_max],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}

emt_sol = solve_ivp(emt_dynamics, y0=x0, **settings)

# Define timepoints that will be used to evaluate the solution of the ODEs
tps = np.linspace(0, t_max, 1000)
n_tps = len(tps)
interp_sol = emt_sol.sol(tps)
angle, i_d, i_q, i_0, i_fd, i_1d, i_1q, i_2q = interp_sol

emtdata = {'time': tps, 'i_d': i_d, 'i_q': i_q, 'i_fd': i_fd, 'i_1d': i_1d, 'i_1q': i_1q, 'i_2q': i_2q}
emtdata = pl.DataFrame(emtdata)

print(emtdata)

matdata = pl.read_csv(f"/Users/psernatorre/Documents/MATLAB/testing_sm/test.csv", has_header=False,
                      new_columns=["time", "i_q", "i_d", "i_fd", "i_1q", "i_2q",  "i_1d"])
print(matdata)

fig, ax = plt.subplots(3, 2, sharex=True)

axs = ax.flatten()

for df in [emtdata, matdata]:
 
    for i, col in enumerate(["i_d", "i_q", "i_fd", "i_1d", "i_1q", "i_2q"]):
        axs[i].set_ylabel(col)
        axs[i].plot(df['time'], df[col])

plt.legend()
plt.show()