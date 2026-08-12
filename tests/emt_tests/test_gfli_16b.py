import os

import matplotlib
import numpy as np
import polars as pl
import pylab as plt

from sting import datasets, main
from sting.generator import GFLI16B
from sting.generator.core import PowerFlowVariables
from sting.utils.dynamical_systems import smooth_step

matplotlib.use('TkAgg')

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)


gfli_1 = GFLI16B(
    name="gfli_1", bus="bus_2",
    # Power flow 
    minimum_active_power_MW=-100, maximum_active_power_MW=-50, minimum_reactive_power_MVAR=-100, maximum_reactive_power_MVAR=100,
    cost_variable_USDperMWh=10, base_power_MVA=100, base_voltage_kV=0.48, base_frequency_Hz=60,
    # LCL filter
    rf1_pu=0.002, xf1_pu=0.07, csh_pu=0.01, rsh_pu=1, 
    txr_power_MVA=100, txr_voltage1_kV=0.48, txr_voltage2_kV=230, txr_r1_pu=0.003/2, txr_x1_pu=0.08/2, txr_r2_pu=0.003/2, txr_x2_pu=0.08/2, 
    # Phase-locked loop (PLL)
    kp_pll_rad_s=100, ki_pll_rad2_s2=2500, tau_pll_s=1/100,
    # Inner current controller
    kp_cc_pu=0.05, ki_cc_puHz=0.6, kff_cc=0.75,
    # Power controllers
    kp_pc_pu=0.1, ki_pc_puHz=200, alpha_pll=0
)

# --------------------------------------------------------
# Check that the linearized QBM and SSM match 
# --------------------------------------------------------
gfli_1.power_flow_variables = PowerFlowVariables(
    p_bus=-1.0000000099749409, 
    q_bus=0.490865108361813, 
    vmag_bus=1.0097648817014873, 
    vphase_bus=-8.816960691471156)

gfli_1._calculate_emt_initial_conditions()
gfli_1._build_small_signal_model()
gfli_1._build_quadratic_bilinear_model()

# Check that initial conditions are an equilibrium
x0 = gfli_1.qbm.x.init
u0 = gfli_1.qbm.u.init
assert np.isclose(gfli_1.qbm.get_derivatives_step(0, x0, lambda t: u0), 0).all()

# Shift the QBM to the initial conditions
import control as ct
ssm = ct.ss(gfli_1.ssm.A, gfli_1.ssm.B,gfli_1.ssm.C,gfli_1.ssm.D)
qbm = gfli_1.qbm.shift_to_equilibrium()
qbm = ct.ss(qbm.A, qbm.B, qbm.C, qbm.D)

# Create a sigma plot
omg = [1e-2, 1e5]
ct.singular_values_plot(ssm, omega_limits=omg)
ct.singular_values_plot(qbm, color="C1", ls="--", omega_limits=omg)
plt.show()

# Create an eigenvalue plot
ssm_ev = np.linalg.eigvals(ssm.A)
qbm_ev = np.linalg.eigvals(qbm.A)

plt.scatter(ssm_ev.real, ssm_ev.imag)
plt.scatter(qbm_ev.real, qbm_ev.imag, marker="x")
plt.xscale('symlog')
plt.yscale('symlog')
plt.show()

# -------------------------------------------------------
# Run  EMT simulations
# -------------------------------------------------------

# Toy 2 bus system
gfli_1.power_flow_variables = None

system = datasets.toy_2(case_directory=case_directory)
system.add(gfli_1)
system.apply("post_system_init", system)

# Create a QBM and SSM
sys, qbm = main.run_qbm(case_directory=case_directory, system=system)
_, ssm = main.run_ssm(case_directory, system=system)

# Check that initial conditions are an equilibrium
x0 = qbm.x.init
u0 = qbm.u.init
assert np.isclose(qbm.get_derivatives_step(0, x0, lambda t: u0), 0, atol=1e-6).all()

# Simulation inputs
inputs = {
    'gfli_16b_0': {
        'q_ref': lambda t: smooth_step(t, step_time=0.2, initial_value=0.0, final_value=0.5, transient_width=5e-3),
        'p_ref': lambda t: smooth_step(t, step_time=0.2, initial_value=0.0, final_value=-0.5, transient_width=5e-3),
        },
}

t_max = 1.5 # Simulation length in seconds

# Small-signal
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Quadratic bilinear
qbm_sol = qbm.simulate(t_max=t_max, inputs=inputs, shift=False)
os.makedirs(os.path.join(case_directory, "outputs", "quadratic_bilinear"), exist_ok=True)
qbm.write_simulation_plots(qbm_sol, os.path.join(case_directory, "outputs", "quadratic_bilinear"))
qbm.write_simulation_csv(qbm_sol, os.path.join(case_directory, "outputs", "quadratic_bilinear"))
# Nonlinear EMT 
main.run_emt(t_max, inputs, case_directory, system=system)

# Compare emt vs ssm results
emt_results = pl.read_csv(f"{case_directory}/outputs/simulation_emt/gfli_16b_0.csv")
qbm_results = pl.read_csv(f"{case_directory}/outputs/quadratic_bilinear/gfli_16b_0.csv").rename({"i_br_d": "i_vsc_d", "i_br_q":"i_vsc_q"})
ssm_results = pl.read_csv(f"{case_directory}/outputs/small_signal_model/gfli_16b_0.csv")


fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 6), sharex=True)
ls = ["-", "--", "-."]

for ax, col in zip(axs.flatten(), ["v_pll_q", "z_apc", "z_rpc", "z_cc_d", "z_cc_q", "i_vsc_d", "i_vsc_q"]):
    i = 0
    for name, df in zip(["EMT", "QBM", "SSM"], [emt_results, qbm_results, ssm_results]):
        ax.plot(df["time"], df[col], label=name, ls=ls[i], color=f"C{i}")
        i += 1
    ax.set_ylabel(col)

ax.legend()
plt.show()

print("ok")