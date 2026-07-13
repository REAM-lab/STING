from sting.components import InnerCurrentController

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pylab as plt

from scipy.integrate import solve_ivp

# Initial conditions
init = {
    'v_vsc_d': 1.215,
    'v_vsc_q': 0.043,
    'v_d': 1.2, 
    'v_q': 0, 
    'i_d': 0.6667, 
    'i_q':-0.435
}


# Simulation inputs (relative to the steady state values)
inputs = {
    "i_d_ref": lambda t: 0.01 if t > 0.5 else 0.0,
    "i_q_ref": lambda t: -0.01 if t > 0.5 else 0.0,
    "i_d": lambda t: 0.02 if t > 0.5 else 0.0,
    "i_q": lambda t: -0.02 if t > 0.5 else 0.0,
    "v_d": lambda t: 0.01 if t > 0.5 else 0.0,
    "v_q": lambda t: 0.01 if t > 0.5 else 0.0,
}

wbase = 2*np.pi*60
# LCL filter model
cc = InnerCurrentController(kp=5, ki=10, kff=0.75, xf=0.02)

# Compute initial conditions and small signal model 
x0 = np.array(cc.get_steady_state(**init))
u0 = np.array([init["i_d"], init["i_q"], init["i_d"], init["i_q"], init["v_d"], init["v_q"]])
y0 = np.array([init["v_vsc_d"], init["v_vsc_q"]])

ssm = cc.get_small_signal_model(*x0)

def ssm_dynamics(t, x):
    """Wrapper function for ODE simulation step"""
    u = np.array([u(t) for u in inputs.values()])
    return ssm.A @ x + ssm.B @ u

def ssm_algebraic(t, x):
    u = np.array([u(t) for u in inputs.values()])
    return ssm.C @ x + ssm.D @ u

def emt_dynamics(t, x):
    """Wrapper function for ODE simulation step"""
    i_d_ref, i_q_ref, i_d, i_q, v_d, v_q = np.array([u(t) for u in inputs.values()]) + u0
    dx = cc.differential_step_emt_dq0(i_d_ref, i_q_ref, i_d, i_q)
    return dx

def emt_algebraic(t, x):
    z_cc_d, z_cc_q = x
    i_d_ref, i_q_ref, i_d, i_q, v_d, v_q = np.array([u(t) for u in inputs.values()]) + u0
    y = cc.algebraic_step_emt_dq0(z_cc_d, z_cc_q, i_d_ref, i_q_ref, i_d, i_q, v_d, v_q)
    return y

# Solve
settings = {
    "t_span": [0,1],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}

emt_sol = solve_ivp(emt_dynamics, y0=x0, **settings)
ssm_sol = solve_ivp(ssm_dynamics, y0=x0*0, **settings)

# TODO: Fix the shapes here
emt_sol.y = np.array([emt_algebraic(emt_sol.t[i], emt_sol.y[:, i]) for i in range(len(emt_sol.t))]).T
ssm_sol.y = np.array([ssm_algebraic(ssm_sol.t[i], ssm_sol.y[:, i]) for i in range(len(ssm_sol.t))]).T + y0.reshape(-1, 1)

# Plot results
titles = [r"$v^{vsc}_d$", r"$v^{vsc}_q$"]
fig, axs = plt.subplots(2, 1)
labels = ["EMT", "SSM"]
ls = ["-", "-."]

for j, sol in enumerate([emt_sol, ssm_sol]):
    for i, ax in enumerate(axs.flatten()):
        ax.set_ylabel(titles[i])
        ax.plot(sol.t, sol.y[i], label=labels[j], ls=ls[j])

plt.legend()
plt.show()