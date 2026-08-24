import matplotlib

from sting.components import VoltageDroopController1A

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

init['q'] = -init['v_d']*init['i_q'] + init['v_q']*init['i_d']

# Simulation inputs (relative to the steady state values)
delta_inputs = {
    "q_ref": lambda t: -0.20 if t > 0.5 else 0.0,
    "v_ref": lambda t: 0.1 if t > 0.5 else 0.0,
    "i_d": lambda t: 0.1 if t > 0.5 else 0.0,
    "i_q": lambda t: -0.5 if t > 0.5 else 0.0,
    "v_d": lambda t: 0.01 if t > 0.5 else 0.0,
    "v_q": lambda t: 0.00 if t > 0.5 else 0.0,
}

ssm_inputs = {
    "q_ref": lambda t: delta_inputs["q_ref"](t),
    "v_ref": lambda t: delta_inputs["v_ref"](t),
    "i_d": lambda t: delta_inputs["i_d"](t) ,
    "i_q": lambda t: delta_inputs["i_q"](t) ,
    "v_d": lambda t: delta_inputs["v_d"](t),
    "v_q": lambda t: delta_inputs["v_q"](t) ,
}

derivative_emt_inputs = {
    "q": lambda t: - delta_inputs['v_d'](t) * delta_inputs['i_q'](t) + delta_inputs['v_q'](t) * delta_inputs['i_d'](t) + init['q']}

algebraic_emt_inputs = {
    "q_ref": lambda t: delta_inputs["q_ref"](t) + init['q'],
    "v_ref": lambda t: delta_inputs["v_ref"](t) + init['v_d'],
}

# Model
mod = VoltageDroopController1A(k_q_pu=0.02, w_q_puHz=50)
ssm = mod.get_small_signal_model(i_d = init['i_d'], i_q = init['i_q'], v_d = init['v_d'], v_q = init['v_q'], q_ref = init['q'], v_ref = init['v_d'])

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
    q_f = x[0]
    q = derivative_emt_inputs["q"](t)
    dx = mod.get_derivatives_step_emt_dq0(q, q_f)
    return dx

def emt_algebraic(t, x):
    q_f = x[0]
    q_ref = algebraic_emt_inputs["q_ref"](t)
    v_ref = algebraic_emt_inputs["v_ref"](t)
    y = mod.get_algebraics_step_emt_dq0(v_ref=v_ref, q_ref=q_ref, q_f=q_f)
    return y

# Solve
settings = {
    "t_span": [0,1],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}

emt_sol = solve_ivp(emt_dynamics, y0=[init['q']], **settings)
ssm_sol = solve_ivp(ssm_dynamics, y0=[0], **settings)

emt_sol.y = np.array([emt_algebraic(emt_sol.t[i], emt_sol.y[:, i]) for i in range(len(emt_sol.t))]).T
ssm_sol.y = np.array([ssm_algebraic(ssm_sol.t[i], ssm_sol.y[:, i]) for i in range(len(ssm_sol.t))]).T + ssm.y.init.reshape(-1, 1)

# Plot results
titles = [r"$q_d$", r"$v^{out}_q$"]
fig, axs = plt.subplots(1, 2)
labels = ["EMT", "SSM"]
ls = ["-", "--"]

for j, sol in enumerate([emt_sol, ssm_sol]):
    for i, ax in enumerate(axs.flatten()):
        ax.set_ylabel(titles[i])
        ax.plot(sol.t, sol.y[i], label=labels[j], ls=ls[j])

plt.legend()
plt.show()