from sting.components import LCLFilter6A

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pylab as plt

from scipy.integrate import solve_ivp

# Power flow solution 
pf_sol = {
    "v_bus_mag": 1.2, # pu
    "relative_phase_deg": 17.2, # deg
    "p_bus": 0.80,  # pu
    "q_bus": 0.522  # pu
}

# Simulation inputs (relative to the steady state values)
inputs = {
    "v_vsc_d": lambda t: 0.01 if t > 0.5 else 0.0,
    "v_vsc_q": lambda t: -0.01 if t > 0.5 else 0.0,
    "v_bus_d": lambda t: 0.02 if t > 0.5 else 0.0,
    "v_bus_q": lambda t: -0.02 if t > 0.5 else 0.0,
    "w": lambda t: 0.01 if t > 0.5 else 0.0,
}

wbase = 2*np.pi*60
# LCL filter model
lcl = LCLFilter6A(rf1_pu=0.001, xf1_pu=0.02, rf2_pu=0.001, xf2_pu=0.01, csh_pu=0.001, rsh_pu=1, wbase=wbase)

# Compute initial conditions and small signal model 
init = lcl.get_steady_state(**pf_sol)
x0 = np.array([init.i_vsc_d, init.i_vsc_q, init.i_bus_d, init.i_bus_q, init.v_sh_d, init.v_sh_q])
u0 = np.array([init.v_vsc_d, init.v_vsc_q, init.v_bus_d, init.v_bus_q, wbase])
ssm = lcl.get_small_signal_model(*x0)


def ssm_dynamics(t, x):
    u = np.array([u(t) for u in inputs.values()])
    return ssm.A @ x + ssm.B @ u


def emt_dynamics(t, x):
    """Wrapper function for ODE simulation step"""
    i_vsc_d, i_vsc_q, i_bus_d, i_bus_q, v_sh_d, v_sh_q = x
    v_vsc_d, v_vsc_q, v_bus_d, v_bus_q, w = np.array([u(t) for u in inputs.values()]) + u0

    dx = lcl.differential_step_emt_dq0(
        i_vsc_d, i_vsc_q, i_bus_d, i_bus_q, v_sh_d, v_sh_q,
        v_vsc_d, v_vsc_q, v_bus_d, v_bus_q, w
    )
    return dx

# Solve
settings = {
    "t_span": [0,1],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}

emt_sol = solve_ivp(emt_dynamics, y0=x0, **settings)
ssm_sol = solve_ivp(ssm_dynamics, y0=x0*0, **settings)
ssm_sol.y += x0.reshape(-1, 1)


# Plot results
titles = [r"$i^{vsc}_d$", r"$i^{vsc}_q$", r"$i^{bus}_d$", r"$i^{bus}_q$", r"$v^f_d$", r"$v^f_q$"]
fig, axs = plt.subplots(2, 3)
labels = ["EMT", "SSM"]
ls = ["-", "-."]

for j, sol in enumerate([emt_sol, ssm_sol]):#
 
    for i, ax in enumerate(axs.flatten()):
        ax.set_ylabel(titles[i])
        ax.plot(sol.t, sol.y[i], label=labels[j], ls=ls[j])

plt.legend()
plt.show()