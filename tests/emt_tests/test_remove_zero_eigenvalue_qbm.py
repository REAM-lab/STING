import os

import numpy as np
import polars as pl
import pylab as plt
from plotly.subplots import make_subplots

from sting import datasets, main
from sting.generator import GFMI18A
from sting.utils.dynamical_systems import DynamicalVariables, QuadraticBilinearModel
from sting.utils.plotting_tools import plot_eigenvalues, compare_timeseries
from sting.utils.matrix_tools import coordinates_to_matrix
# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# -------------------------------------------------------
# Construct a 2-bus system without a slack generator
# -------------------------------------------------------
gfmi2 = GFMI18A(
    name="gfmi_2", bus="bus_2",
    # Power flow 
    minimum_active_power_MW=80, maximum_active_power_MW=80, minimum_reactive_power_MVAR=50, maximum_reactive_power_MVAR=51,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.005, xf1_pu=0.15, csh_pu=0.066, rsh_pu=10,
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.01, txr_x1_pu=0.1, txr_r2_pu=0.02, txr_x2_pu=0.1, 
    # Inner voltage controller
    kp_vc_pu=0.562, ki_vc_puHz=484.989, kffi_vc=0.80,
    # Inner current controller
    kp_cc_pu=4.77, ki_cc_puHz=60, kffv_cc=0,
    # Virtual inertia
    h_s=2, kd_pu=70, alpha=1,
    # Voltage droop
    k_q_pu=0.2, w_q_puHz=4000
)

gfmi1 = GFMI18A(
    name="gfmi_1", bus="bus_1",
    # Power flow 
    minimum_active_power_MW=-200, maximum_active_power_MW=200, minimum_reactive_power_MVAR=-500, maximum_reactive_power_MVAR=500,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.005, xf1_pu=0.15, csh_pu=0.066, rsh_pu=10,
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.01, txr_x1_pu=0.1, txr_r2_pu=0.02, txr_x2_pu=0.1, 
    # Inner voltage controller
    kp_vc_pu=0.562, ki_vc_puHz=484.989, kffi_vc=0.80,
    # Inner current controller
    kp_cc_pu=4.77, ki_cc_puHz=60, kffv_cc=0,
    # Virtual inertia
    h_s=2, kd_pu=70, alpha=1,
    # Voltage droop
    k_q_pu=0.2, w_q_puHz=4000
)
system = datasets.toy_2(case_directory=case_directory)
system.add(gfmi1)
system.add(gfmi2)
system.voltage_source_4a.clear()
system.apply("post_system_init", system)

# -------------------------------------------------------
# Construct the QBM about an equilibrium
# -------------------------------------------------------
_, qbm = main.run_qbm(case_directory, system=system)
qbm_shifted = qbm.shift_to_equilibrium()
print("Original EVs")
print(np.sort(np.linalg.eigvals(qbm_shifted.A)))

# -------------------------------------------------------
# Close the loop
# -------------------------------------------------------

def remove_zero_eigenvalue(qbm:QuadraticBilinearModel, slack_generator:str, drop=True):
    # State, inputs, and outputs
    x, u, y = qbm.x, qbm.u, qbm.y
    n, m, p = len(x), len(u), len(y)

    # Mask arrays
    is_slack_input = (u.name == 'w_slack')
    is_slack_component = (x.component == slack_generator)
    is_w_state = (x.name == 'w')
    is_sin_state = (x.name == 'sin')
    is_cos_state = (x.name == 'cos')
    is_phase_state = is_sin_state | is_cos_state

    # Modify outputs to include the slack angular velocity
    C_slack_w = (is_w_state & is_slack_component).astype(int).reshape(-1,1)
    C = np.vstack([qbm.C, C_slack_w.T])
    D = np.vstack([qbm.D, np.zeros((1,m))])
    qbm_extended = QuadraticBilinearModel(
        A=qbm.A, B=qbm.B, C=C, D=D, N=qbm.N, H=qbm.H, u=qbm.u, x=qbm.x,
        y=DynamicalVariables(name=list(y.name)+['w_slack']))

    # Interconnection matrices #
    u_sys = u[~is_slack_input]
    # Close the loop from outputs to inputs
    L11 = np.hstack([np.zeros((m, p)), is_slack_input.astype(int).reshape(-1,1)])
    # Delete all w_slack inputs
    L12 = np.eye(m)[:,~is_slack_input]
    L21 = np.eye(N=p, M=p+1)
    L22 = np.zeros((p,len(u_sys)))
    qbm_closed = QuadraticBilinearModel.from_interconnected([qbm_extended], [L11,L12,L21,L22,None,None], u=u_sys, y=y)

    # Compute new angle initial conditions
    J = np.array([[0,1],[-1,0]])
    I = np.eye(2)
    # Assuming every generator component has sin and cos pair. By trigonometry
    #   sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
    #   cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
    phase_pairs = np.hsplit(np.arange(n)[is_phase_state], 2)
    W2_sin = coordinates_to_matrix(shape=(n,n), data=[(pair,pair,-J) for pair in phase_pairs])
    W2_cos = coordinates_to_matrix(shape=(n,n), data=[(pair,pair,I) for pair in phase_pairs])
    sin0_slack = (x.init[(is_sin_state & is_slack_component)]).item()
    cos0_slack = (x.init[(is_cos_state & is_slack_component)]).item()
    # All other states have the same initial conditions
    W1 = np.diag((~is_phase_state).astype(int))
    # Save the new initial conditions
    x0 = (W1 + W2_sin*sin0_slack + W2_cos*cos0_slack)@x.init
    #qbm_closed.x.init = x0

    if drop:

        qbm_closed = qbm_closed.shift_to_equilibrium()

        # Transformation matrix to drop the slack phase angle states
        is_slack_phase_state = (is_phase_state & is_slack_component)
        T = np.eye(n)[~is_slack_phase_state, :]
        # Remove the states corresponding to the slack generator
        qbm_reduced = qbm_closed.project(
            W=T, V=T.transpose(), 
            name=x.name[~is_slack_phase_state], 
            component=x.component[~is_slack_phase_state])

        return qbm_reduced

    return qbm_closed


print("New EVs")
qbm_r = remove_zero_eigenvalue(qbm, "gfmi_18a_0", drop=False)
#print(np.sort(np.linalg.eigvals(qbm_r.A)))

# -------------------------------------------------------
# Plot and compare eigenvalues
# -------------------------------------------------------

fig = make_subplots(rows=1, cols=1)
fig = plot_eigenvalues(fig, qbm_shifted.A)
fig = plot_eigenvalues(fig, qbm_r.A, marker_color="red", marker_symbol="triangle-up")

fig.write_html(os.path.join(case_directory, "eigenvalues.html"))


# -------------------------------------------------------
# Simulate dynamics
# -------------------------------------------------------

from sting.utils.dynamical_systems import make_smooth_step

inputs = { 
    'gfmi_18a_0': {
        'v_ref': make_smooth_step(step_time=0.10, initial_value=0.0, final_value=0.50, transient_width=5e-3),
        'p_ref': make_smooth_step(step_time=0.10, initial_value=0.0, final_value=0.50, transient_width=5e-3),
        'q_ref': make_smooth_step(step_time=0.10, initial_value=0.0, final_value=-0.50, transient_width=5e-3) 
        }
}
t_max = 1.5 # Simulation length in seconds

os.makedirs(os.path.join(case_directory, "outputs", "quadratic_bilinear_1"), exist_ok=True)
os.makedirs(os.path.join(case_directory, "outputs", "quadratic_bilinear_2"), exist_ok=True)
# QBM 
_, qbm = main.run_qbm(case_directory, system=system)
sol = qbm.simulate(t_max=t_max, inputs=inputs)
qbm.write_simulation_csv(sol, os.path.join(case_directory, "outputs", "quadratic_bilinear_1"))

sol = qbm_r.simulate(t_max=t_max, inputs=inputs)
qbm_r.write_simulation_csv(sol, os.path.join(case_directory, "outputs", "quadratic_bilinear_2"))

# -------------------------------------------------------
# Plot and compare
# -------------------------------------------------------

# Compare the results of the EMT and small-signal model simulations
file = "gfmi_18a_0.csv" #"branch_series_rl_0.csv"#
cols = ["w", "q_f", "z_vc_d", "z_vc_q", "z_cc_d", "z_cc_q", "i_br_d", "i_br_q", "i_br_D", "i_br_Q", "v_sh_D", "v_sh_Q"]
#cols = ["i_br_D", "i_br_Q"]

compare_timeseries(
    df1=pl.read_csv(f"{case_directory}/outputs/quadratic_bilinear_1/{file}"),
    df2=pl.read_csv(f"{case_directory}/outputs/quadratic_bilinear_2/{file}"),
    left_to_right=dict(zip(cols, cols)),
    df1_name="QBM1",
    df2_name="QBM2",
    figure_filepath=f"{case_directory}/outputs/comparison_{file}.html",
    df1_color="blue",
    df2_color="red"
)
