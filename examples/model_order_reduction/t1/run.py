"""
Testcase1 simulates a two infinite sources connected via a transmission line.

First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.

You should obtain the following eigenvalues:
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|    |   Eigenvalue  |       Eigenvalue  |        Damping  |          Natural  |           Time  |
|    |     real part |    imaginary part |    ratio [p.u.] |    frequency [Hz] |    constant [s] |
+====+===============+===================+=================+===================+=================+
|  0 |       -21.965 |           376.991 |           0.058 |            60.102 |           0.046 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|  1 |       -21.965 |          -376.991 |           0.058 |            60.102 |           0.046 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|  2 |      -159.588 |          2149.008 |           0.074 |           342.967 |           0.006 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|  3 |      -159.588 |         -2149.008 |           0.074 |           342.967 |           0.006 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|  4 |      -159.588 |          2902.990 |           0.055 |           462.723 |           0.006 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|  5 |      -159.588 |         -2902.990 |           0.055 |           462.723 |           0.006 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|  6 |      -166.837 |          3750.794 |           0.044 |           597.548 |           0.006 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|  7 |      -166.837 |         -3750.794 |           0.044 |           597.548 |           0.006 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|  8 |      -166.837 |          4504.776 |           0.037 |           717.449 |           0.006 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
|  9 |      -166.837 |         -4504.776 |           0.037 |           717.449 |           0.006 |
+----+---------------+-------------------+-----------------+-------------------+-----------------+
"""

# Import Python standard and third-party packages
from pathlib import Path
import pylab as plt
from control import singular_values_plot

# Import sting package
from sting import main
from sting.modules.model_order_reduction.reductions import SingularPerturbation, BalancedTruncation

import os

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

reductions = {
    "zone_1":  SingularPerturbation(r=4, basis="eigen")
}

ssm, fom, rom_mr = main.run_model_reduction(case_directory=case_dir, reductions=reductions)


reductions = {
    "zone_1":  BalancedTruncation(r=4, gramian_c="subsystem", gramian_o="subsystem", method="truncate")
}

_, _, rom_br = main.run_model_reduction(case_directory=case_dir, reductions=reductions)



red = "#BB5566"
yellow = "#DDAA33"
dark_blue = "#004488"
light_blue = "#6699CC"

# Compare the eigenvalues of the FOM and ROM
ax = fom.plot_eigenvalues(marker="x", label="Full-order model", color="gray")
ax = rom_mr.plot_eigenvalues(ax=ax, marker="^", label="Modal Reduction", color=red)
ax = rom_br.plot_eigenvalues(ax=ax, marker="o", label="Balanced Reduction", color=light_blue)
ax.set_xscale("symlog")
ax.legend()

plt.savefig(os.path.join(case_dir, "outputs", "eigenvalues.pdf"))

# Compare the sigmaplots of the FOM and ROM
singular_values_plot(fom.to_python_control(), label="Full-order model", color="gray", omega=[1e1, 1e4])
singular_values_plot(rom_mr.to_python_control(), label="Modal Reduction", color=red, ls="--", omega=[1e1, 1e4])
singular_values_plot(rom_br.to_python_control(), label="Balanced Reduction", color=light_blue, ls="--", omega=[1e1, 1e4])
plt.ylim(1e-1, 1e2)
plt.savefig(os.path.join(case_dir, "outputs", "sigmaplot.pdf"))

print('ok')