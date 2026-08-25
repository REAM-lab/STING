import os

import h5py
import matplotlib
import numpy as np
import pylab as plt
from scipy.integrate import solve_ivp

from sting import datasets, main
from sting.components import ExcitationSystem4A, VoltageTransducer1A

matplotlib.use('TkAgg')

transducer = VoltageTransducer1A(tau_s=5e-2)
exciter = ExcitationSystem4A(
    ka_pu=300, ta_s=0.001,
    ke_pu=1, te_s=1.15,
    tb_s=0.006, tc_s=0.173,
    kf_pu=0.001, tf_s=0.1,
)

inputs = {
    "v_ref": lambda t: 1.28 if t < 1 else 1.28+ 0.001*np.sin(t*5),
    "v_d": lambda t: 1.28 if t < 1 else 1.3,
    "v_q": lambda t: 0 if t < 1 else 0.04,
}

u0 = np.array([u(0) for _, u in inputs.items()])

transducer.get_steady_state(v_d=inputs["v_d"](0), v_q=inputs["v_q"](0))
exciter.get_steady_state(v_ref=inputs["v_ref"](0), v_c=transducer.emt_init.v_c1, v_s=0)

t_ssm = transducer.get_small_signal_model(
    v_d=inputs["v_d"](0), 
    v_q=inputs["v_q"](0), 
)

v_mag = transducer.emt_init.v_c1
e_ssm = exciter.get_small_signal_model(
    x_l=exciter.emt_init.x_l,
    x_a=exciter.emt_init.x_a,
    x_e=exciter.emt_init.x_e,
    x_f=exciter.emt_init.x_f,
    v_ref=inputs["v_ref"](0),
    v_c=v_mag,
    v_s=0
)
t_qbm = transducer.get_quadratic_bilinear_model(v_d=inputs["v_d"](0), v_q=inputs["v_q"](0))
c0, c1, c2 = transducer.get_taylor_series_constants(v_mag=v_mag)
def emt_dynamics(t, x):
    v_c1, x_l, x_a, x_e, x_f = x 
    v_ref, v_d, v_q = [u(t) for _, u in inputs.items()]
    dx1 = transducer.get_derivatives_step_emt_dq0(v_c1=v_c1, v_d=v_d, v_q=v_q)
    dx2 = exciter.get_derivatives_step_emt_dq0(x_l, x_a, x_e, x_f, v_ref, v_c1, 0)

    return np.concat([dx1, dx2])

def qbm_dynamics(t, x):
    v_c1, x_l, x_a, x_e, x_f = x 
    v_ref, v_d, v_q = [u(t) for _, u in inputs.items()]
    dx1 = t_qbm.A@np.array([v_c1]) + t_qbm.B@np.array([v_d**2, v_q**2])
    v_c = c0 + c1*v_c1 + c2*v_c1**2
    dx2 = exciter.get_derivatives_step_emt_dq0(x_l, x_a, x_e, x_f, v_ref, v_c, 0)

    return np.concat([dx1, dx2])

def ssm_dynamics(t, x):
    v_c1, x_l, x_a, x_e, x_f = x 
    v_ref, v_d, v_q = np.array([u(t) for _, u in inputs.items()]) - u0

    dx1 = t_ssm.A@np.array([v_c1]) + t_ssm.B@np.array([v_d, v_q])
    dx2 = e_ssm.A@np.array([x_l, x_a, x_e, x_f]) + e_ssm.B@np.array([v_ref, v_c1, 0])

    return np.concat([dx1, dx2])


y0 = np.array([
    transducer.emt_init.v_c1, 
    exciter.emt_init.x_l, 
    exciter.emt_init.x_a,
    exciter.emt_init.x_e, 
    exciter.emt_init.x_f
    ])

settings = {
    "t_span": [0,10],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}

emt_sol = solve_ivp(emt_dynamics, y0=y0, **settings)
emt_sol.y[3] += inputs["v_ref"](0)

y0_qbm = y0.copy()
y0_qbm[0] *= y0_qbm[0]
qbm_sol = solve_ivp(qbm_dynamics, y0=y0_qbm, **settings)
qbm_sol.y[3] += inputs["v_ref"](0)
qbm_sol.y[0] = qbm_sol.y[0]**0.5

ssm_sol = solve_ivp(ssm_dynamics, y0=y0*0, **settings)
ssm_sol.y += y0.reshape(-1, 1)
ssm_sol.y[3] += inputs["v_ref"](0)

# Load MATLAB solution: You will need to run the associated simulink file.
file_path = os.path.join(os.getcwd(), "tests", "emt_tests", "matlab_data.mat")
with h5py.File(file_path, 'r') as file:
    # Check the variable names
    print(list(file.keys()))
    
    data = file['ans'][:]

# Plot results
titles = [r"$v_m$", r"$x_l$", r"$x_a$", r"$x_e$", r"$x_f$",]
fig, axs = plt.subplots(2, 3)
axs = axs.flatten()
labels = ["EMT", "QBM", "SSM"]
ls = ["-", "-.", "--"]

for j, sol in enumerate([emt_sol, qbm_sol, ssm_sol]):
 
    for i, ax in enumerate(axs):
        if i == 5:
            break
        ax.set_ylabel(titles[i])
        ax.plot(sol.t, sol.y[i], label=labels[j], ls=ls[j])

axs[3].plot(data[:,0], data[:,1], ls='--')

plt.legend()
plt.show()


print("ok")