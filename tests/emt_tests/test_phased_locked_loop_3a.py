import matplotlib

from sting.components import PhaseLockedLoop3A

matplotlib.use('TkAgg')
import numpy as np
import pylab as plt
from scipy.integrate import solve_ivp

# Power flow solution 
pf_sol = {
    "v_mag": 1.2, # pu
    "relative_phase_deg": 17.2, # deg
}

# Initial conditions
phase_rad = pf_sol["relative_phase_deg"] * np.pi / 180
v_bus_DQ = pf_sol["v_mag"] * np.exp(phase_rad * 1j)
u0 = np.array([v_bus_DQ.real, v_bus_DQ.imag])

# Simulation inputs (relative to the steady state values)
inputs = {
    "v_bus_D": lambda t: 0.21 if t > 0.5 else 0.0,
    "v_bus_Q": lambda t: -0.51 if t > 0.5 else 0.0,
}

wbase = 2*np.pi*60
alpha = 5
# LCL filter model
pll = PhaseLockedLoop3A(kp_pu=100, ki_puHz=2500, tau=0.01, alpha=alpha, wbase=wbase)

# Compute initial conditions and small signal model 
init = pll.get_steady_state(**pf_sol)
ssm = pll.get_small_signal_model(**pf_sol)
qbm = pll.get_quadratic_bilinear_model(**pf_sol)
qbm.shift_to_equilibrium()

def ssm_dynamics(t, x):
    u = np.array([u(t) for u in inputs.values()])
    return ssm.A @ x + ssm.B @ u

def qbm_dynamics(t, x):
    u = np.array([u(t) for u in inputs.values()])
    return qbm.A @ x + qbm.B @ u + qbm.H @ np.kron(x,x) + qbm.N @ np.kron(u, x)

def emt_dynamics(t, x):
    """Wrapper function for ODE simulation step"""
    v_pll_q, z_pll, phase_pll = x
    v_bus_D, v_bus_Q = np.array([u(t) for u in inputs.values()]) + u0

    dx = pll.get_derivatives_step_emt_dq0(
        v_pll_q, z_pll, phase_pll, v_bus_D, v_bus_Q
    )
    return dx

# Solve
settings = {
    "t_span": [0,1],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}

emt_sol = solve_ivp(emt_dynamics, y0=ssm.x.init, **settings)
qbm_sol = solve_ivp(qbm_dynamics, y0=qbm.x.init*0, **settings)
qbm_sol.y += qbm.x.init.reshape(-1, 1)
ssm_sol = solve_ivp(ssm_dynamics, y0=ssm.x.init*0, **settings)
ssm_sol.y += ssm.x.init.reshape(-1, 1)

# Recover angle from QBM
qbm_sol.y[2, :] = np.atan2(qbm_sol.y[2, :], qbm_sol.y[3, :])

# Plot results
titles = [r"$v_q$", r"$\epsilon$", r"$\delta$"]
fig, axs = plt.subplots(1,3)
labels = ["EMT", "QBM", "SSM"]
ls = ["-", "-.", "--"]

for j, sol in enumerate([emt_sol, qbm_sol,  ssm_sol]):
 
    for i, ax in enumerate(axs.flatten()):
        ax.set_ylabel(titles[i])
        ax.plot(sol.t, sol.y[i], label=labels[j], ls=ls[j])

plt.legend()
plt.show()

# Check that there is an eigenvalue at the predicted position
print(np.linalg.eigvals(qbm.A), -2*alpha*(np.sin(phase_rad)+np.cos(phase_rad)))