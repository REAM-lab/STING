import matplotlib

from sting import main
from sting.components import RotationalInertia2A

matplotlib.use('TkAgg')
import numpy as np
import pylab as plt
from scipy.integrate import solve_ivp

# Initial conditions
init = {
    'v_d': 1.0,
    'v_q': 0.0,
    'i_d': 1.25,
    'i_q': 0.0,
}

init['p'] = init['v_d']*init['i_d'] + init['v_q']*init['i_q']

# Simulation inputs (relative to the steady state values)
delta_inputs = {
    "p_ref": lambda t: -0.20 if t > 0.5 else 0.0,
    "i_d": lambda t: 0.1 if t > 0.5 else 0.0,
    "i_q": lambda t: -0.5 if t > 0.5 else 0.0,
    "v_d": lambda t: 0.01 if t > 0.5 else 0.0,
    "v_q": lambda t: 0.00 if t > 0.5 else 0.0,
}

ssm_inputs = {
    "p_ref": lambda t: delta_inputs["p_ref"](t),
    "i_d": lambda t: delta_inputs["i_d"](t) ,
    "i_q": lambda t: delta_inputs["i_q"](t) ,
    "v_d": lambda t: delta_inputs["v_d"](t),
    "v_q": lambda t: delta_inputs["v_q"](t) ,
}

qbm_inputs = {
    "p_ref": lambda t: delta_inputs["p_ref"](t) + init['p'],
    "w_slack": lambda t: 1,
    "one": lambda t:1,
    "p": lambda t: delta_inputs['v_d'](t) * delta_inputs['i_d'](t) + delta_inputs['v_q'](t) * delta_inputs['i_q'](t) + init['p']}

derivative_emt_inputs = {
    "p_ref": lambda t: delta_inputs["p_ref"](t) + init['p'],
    "p": lambda t: delta_inputs['v_d'](t) * delta_inputs['i_d'](t) + delta_inputs['v_q'](t) * delta_inputs['i_q'](t) + init['p']}

# Model
mod = RotationalInertia2A(h_s= 2, kd_w_pu=70, w_nom=2*np.pi*60)
ssm = mod.get_small_signal_model(i_d = init['i_d'], i_q = init['i_q'], v_d = init['v_d'], v_q = init['v_q'], angle = 0, p_ref = init['p'])
qbm = mod.get_quadratic_bilinear_model(w=1, angle_rad=0, p_ref=init['p'], p=init['p'])


# Differential equations
def ssm_dynamics(t, x):
    """Wrapper function for ODE simulation step"""
    u = np.array([u(t) for u in ssm_inputs.values()])
    return ssm.A @ x + ssm.B @ u

def ssm_algebraic(t, x):
    u = np.array([u(t) for u in ssm_inputs.values()])
    return ssm.C @ x + ssm.D @ u

def emt_dynamics(t, x):
    """Wrapper function for ODE simulation step"""
    angle, w = x[0], x[1]
    p = derivative_emt_inputs["p"](t)
    p_ref = derivative_emt_inputs["p_ref"](t)
    dx = mod.get_derivatives_step_emt_abc(w, p_ref, p)
    return dx

def qbm_dynamics(t,x):
    """Wrapper function for ODE simulation step"""
    u = np.array([u(t) for u in qbm_inputs.values()])
    return qbm.A @ x + qbm.B @ u + qbm.H @ np.kron(x,x) + qbm.N @ np.kron(u,x)


# Solve
settings = {
    "t_span": [0,1],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}

emt_sol = solve_ivp(emt_dynamics, y0=[0, 1], **settings)
ssm_sol = solve_ivp(ssm_dynamics, y0=[0, 0], **settings)
qbm_sol = solve_ivp(qbm_dynamics, y0=[1, np.sin(0), np.cos(0)], **settings)

emt_sol.y = np.array([emt_sol.y[:, i] for i in range(len(emt_sol.t))]).T
ssm_sol.y = np.array([ssm_algebraic(ssm_sol.t[i], ssm_sol.y[:, i]) for i in range(len(ssm_sol.t))]).T + ssm.y.init.reshape(-1, 1)


angle = np.atan2(qbm_sol.y[1], qbm_sol.y[2])
w = qbm_sol.y[0]
qbm_sol.y = (angle, w)

# Plot results
titles = [r"angle", r"$\omega$"]
fig, axs = plt.subplots(1, 2)
labels = ["EMT", "QBM", "SSM"]
ls = ["-", "--", "-."]

for j, sol in enumerate([emt_sol, qbm_sol, ssm_sol]):
    for i, ax in enumerate(axs.flatten()):
        ax.set_ylabel(titles[i])
        ax.plot(sol.t, sol.y[i], label=labels[j], ls=ls[j])

plt.legend()
plt.show()