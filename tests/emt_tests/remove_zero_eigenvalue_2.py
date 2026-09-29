import os

import numpy as np
import sympy as sp
from plotly.subplots import make_subplots
from sympy import Matrix
from sympy.physics.quantum import TensorProduct

from sting.utils.dynamical_systems import kronecker_commute
from sting.utils.plotting_tools import plot_eigenvalues

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)


def vectorize(x):
    symbol_creator = np.vectorize(sp.Symbol)
    return symbol_creator(x)

# ------------------------------------------------------------
# System dynamics
# ------------------------------------------------------------
b = 0.1
kd = 0.05
a1 = 1
a2 = 1

A = np.array([
    [ 0, 0,  0, 0, 0, 0],
    [ 0, 0,  0, 0, 0, 0],
    [-b, 0,-kd, b, 0, 0],
    [ 0, 0,  0, 0, 0, 0],
    [ 0, 0,  0, 0, 0, 0],
    [ b, 0,  0,-b, 0,-kd]
])
H_s1 = np.zeros((6,6))
H_s1[1,2] = -1
H_s1[0,0] = -a1
H_s1[1,0] = -a1

H_c1 = np.zeros((6,6))
H_c1[0,2] = 1
H_c1[0,1] = -a1
H_c1[1,1] = -a1

H_s2 = np.zeros((6,6))
H_s2[4,5] = -1
H_s2[3,3] = -a2
H_s2[4,3] = -a2

H_c2 = np.zeros((6,6))
H_c2[3,5] = 1
H_c2[3,4] = -a2
H_c2[4,4] = -a2

Z = np.zeros((6,6))
H = np.hstack([H_s1, H_c1, Z, H_s2, H_c2, Z])

B = np.array([
    [a1, 0, 0],
    [a1, 0, 0],
    [ 0, 0,-1],
    [a2, 0, 0],
    [a2, 0, 0],
    [ 0, 0,-1],
])

N_w = np.array([
    [ 0,-1,  0, 0, 0, 0],
    [ 1, 0,  0, 0, 0, 0],
    [ 0, 0,  0, 0, 0, 0],
    [ 0, 0,  0, 0,-1, 0],
    [ 0, 0,  0, 1, 0, 0],
    [ 0, 0,  0, 0, 0, 0],
])

N = np.hstack([Z, N_w, Z])

C = np.eye(6)

# ------------------------------------------------------------
# Check the dynamics
# ------------------------------------------------------------

x = vectorize(["sin_1",  "cos_1",  "w_1", "sin_2", "cos_2", "w_2"])
u = vectorize(["one", "w_ref", "delta"])

dx = Matrix(A) @ x + Matrix(H)@ TensorProduct(x, x) + Matrix(B)@u + Matrix(N)@ TensorProduct(u, x)

for dx_i in dx:
    print(sp.nsimplify(dx_i))



# ------------------------------------------------------------
# Close the loop
# ------------------------------------------------------------

"""
        │ sin_1  cos_1  w_1  sin_2  cos_2  w_2 │ one  delta
────────┼──────────────────────────────────────┼────
one     │  0    0       0       0       0   0  │ 1
w_ref   │  0    0       1       0       0   0  │ 0
delta   │  0    0       0       0       0   0  │ 1
"""

L_11 = np.zeros((3,6))
L_11[1,2] = 1
L_12 = np.array([[1,0],[0,0],[0,1]])


A_t = A + B@L_11
H_t = H + N@np.kron(L_11, np.eye(6))
B_t = B@L_12
N_t = N@np.kron(L_12, np.eye(6))

# ------------------------------------------------------------
# Linearize
# ------------------------------------------------------------
n, m = B.shape
K1 = kronecker_commute(n,n)
K2 = kronecker_commute(n,m)

def shifted_A(A, H, N, x, u):
    return (
        A 
        + H @ (K1 + np.eye(n**2)) @ np.kron(x, np.eye(n))
        + N @ np.kron(u, np.eye(n))
    )


phase_1 = 0.1
phase_2 = 0.2
print("Radius 1 EV:", -2*a1*(np.sin(phase_1) + np.cos(phase_1)))
print("Radius 2 EV:", -2*a2*(np.sin(phase_2) + np.cos(phase_2)))

x0 = np.array([np.sin(phase_1), np.cos(phase_1), 0, np.sin(phase_2), np.cos(phase_2), 0,]).reshape(-1, 1)
u0 = np.array([1, 0, 0]).reshape(-1, 1)
A_ssm = shifted_A(A, H, N, x0, u0)

print("\nOriginal EVs")
print(np.sort(np.linalg.eigvals(A_ssm)))

print("\nTruncated EVs")
A_t_ssm = shifted_A(A_t, H_t, N_t, x0, np.array([[1],[0]]))
print(np.sort(np.linalg.eigvals(A_t_ssm[2:,2:])))

fig = make_subplots(rows=1, cols=1)
fig = plot_eigenvalues(fig, A_ssm)
fig = plot_eigenvalues(fig, A_t_ssm[2:,2:], marker_color="red", marker_symbol="triangle-up")

fig.write_html(os.path.join(case_directory, "eigenvalues.html"))





# ------------------------------------------------------------
# Check the trajectories
# ------------------------------------------------------------

from scipy.integrate import solve_ivp

u_3 = lambda t: 0 if t < 0.1 else 0.1



def f(t, x):
    u = np.array([1, 0, u_3(t)])
    return A@x + H@np.kron(x,x) + B@u + N@np.kron(u,x)

def g(t, x):
    u = np.array([1, u_3(t)])
    return A_t@x + H_t@np.kron(x,x) + B_t@u + N_t@np.kron(u,x)   


# Solve
settings = {
    "t_span": [0,1],
    "max_step": 0.001,
    "dense_output": True,
    "method": "Radau"
}
x0_g = np.array([np.sin(0), np.cos(0), 0, np.sin(phase_2-phase_1), np.cos(phase_2-phase_1), 0,])

sol_f = solve_ivp(f, y0=x0.flatten(), **settings)
sol_g = solve_ivp(g, y0=x0_g, **settings)

import pylab as plt
import matplotlib

matplotlib.use('TkAgg')

# Plot results
titles =["sin_1",  "cos_1",  "w_1", "sin_2", "cos_2", "w_2"]
fig, axs = plt.subplots(2, 3)
labels = ["f", "g"]
ls = ["-", "-.", "--"]

for j, sol in enumerate([sol_f, sol_g]):
 
    for i, ax in enumerate(axs.flatten()):
        ax.set_ylabel(titles[i])
        ax.plot(sol.t, sol.y[i], label=labels[j], ls=ls[j])

plt.legend()
plt.show()

print("ok")

