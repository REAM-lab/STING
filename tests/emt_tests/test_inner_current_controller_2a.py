import matplotlib

from sting.components import InnerCurrentController2A

matplotlib.use('TkAgg')
import numpy as np
import pylab as plt
from scipy.integrate import solve_ivp

# Initial conditions
init = {
    'v_out_d': 1.215,
    'v_out_q': 0.043,
    'v_d': 1.2, 
    'v_q': 0, 
    'i_d': 0.6667, 
    'i_q':-0.435,
    'w': 2*np.pi*60
}

# Simulation inputs (relative to the steady state values)
inputs = {
    "i_d_ref": lambda t: 0.01 if t > 0.5 else 0.0,
    "i_q_ref": lambda t: -0.01 if t > 0.5 else 0.0,
    "i_d": lambda t: 0.02 if t > 0.5 else 0.0,
    "i_q": lambda t: -0.02 if t > 0.5 else 0.0,
    "v_d": lambda t: 0.01 if t > 0.5 else 0.0,
    "v_q": lambda t: 0.01 if t > 0.5 else 0.0,
    "w": lambda t: -0.2 if t > 0.5 else 0.0,
}
# Special nonlinear inputs for the QB model
qbm_inputs = inputs.copy()
w_func = qbm_inputs.pop('w')
qbm_inputs['w*i_d'] = lambda t: w_func(t) * inputs['i_d'](t)
qbm_inputs['w*i_q'] = lambda t: w_func(t) * inputs['i_q'](t)

# LCL filter models
cc = InnerCurrentController2A(kp_pu=5, ki_puHz=10, kffv=0.75, xf_pu=0.02)
init = cc.get_steady_state(**init)
ssm = cc.get_small_signal_model(init.z_cc_d, init.z_cc_q, init.i_d, init.i_q, init.v_d, init.v_q, init.w)
qbm = cc.get_quadratic_bilinear_model(init.z_cc_d, init.z_cc_q, init.i_d, init.i_q, init.v_d, init.v_q, init.w)

# Differential equations
def ssm_dynamics(t, x):
    """Wrapper function for ODE simulation step"""
    u = np.array([u(t) for u in inputs.values()])
    return ssm.A @ x + ssm.B @ u

def ssm_algebraic(t, x):
    u = np.array([u(t) for u in inputs.values()])
    return ssm.C @ x + ssm.D @ u

def qbm_dynamics(t, x):
    u = np.array([u(t) for u in qbm_inputs.values()])
    return qbm.A @ x + qbm.B @ u + qbm.H @ np.kron(x,x) + qbm.N @ np.kron(u, x)

def qbm_algebraic(t, x):
    u = np.array([u(t) for u in qbm_inputs.values()])
    return qbm.C @ x + qbm.D @ u

def emt_dynamics(t, x):
    """Wrapper function for ODE simulation step"""
    i_d_ref, i_q_ref, i_d, i_q, v_d, v_q, w = np.array([u(t) for u in inputs.values()]) + ssm.u.init
    dx = cc.get_derivatives_step_emt_dq0(i_d_ref, i_q_ref, i_d, i_q)
    return dx

def emt_algebraic(t, x):
    z_cc_d, z_cc_q = x
    i_d_ref, i_q_ref, i_d, i_q, v_d, v_q, w = np.array([u(t) for u in inputs.values()]) + ssm.u.init
    y = cc.get_algebraics_step_emt_dq0(z_cc_d, z_cc_q, i_d_ref, i_q_ref, i_d, i_q, v_d, v_q, w)
    return y

# Solve
settings = {
    "t_span": [0,1],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}

emt_sol = solve_ivp(emt_dynamics, y0=ssm.x.init, **settings)
qbm_sol = solve_ivp(qbm_dynamics, y0=qbm.x.init, **settings)
ssm_sol = solve_ivp(ssm_dynamics, y0=ssm.x.init*0, **settings)

emt_sol.y = np.array([emt_algebraic(emt_sol.t[i], emt_sol.y[:, i]) for i in range(len(emt_sol.t))]).T
qbm_sol.y = np.array([emt_algebraic(qbm_sol.t[i], qbm_sol.y[:, i]) for i in range(len(qbm_sol.t))]).T
ssm_sol.y = np.array([ssm_algebraic(ssm_sol.t[i], ssm_sol.y[:, i]) for i in range(len(ssm_sol.t))]).T + ssm.y.init.reshape(-1, 1)

# Plot results
titles = [r"$v^{out}_d$", r"$v^{out}_q$"]
fig, axs = plt.subplots(1, 2)
labels = ["EMT", "QBM", "SSM"]
ls = ["-", "-.", "--"]

for j, sol in enumerate([emt_sol, qbm_sol, ssm_sol]):
    for i, ax in enumerate(axs.flatten()):
        ax.set_ylabel(titles[i])
        ax.plot(sol.t, sol.y[i], label=labels[j], ls=ls[j])

plt.legend()
plt.show()