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
from sting.modules.model_order_reduction.core import SingularPerturbation

import os

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

reductions = {
    "zone_1":  SingularPerturbation(r=4, basis="eigen")
}

fom, rom = main.run_model_reduction(case_directory=case_dir, reductions=reductions)


# Compare the eigenvalues of the FOM and ROM
ax = fom.model.plot_eigenvalues(marker="o", label="FOM")
ax = rom.model.plot_eigenvalues(ax=ax, marker="^", label="ROM", color="r")
ax.set_xscale("symlog")
ax.legend()

plt.savefig(os.path.join(case_dir, "outputs", "eigenvalues.pdf"))

# Compare the sigmaplots of the FOM and ROM
singular_values_plot(fom.model.to_python_control(), label="FOM", omega=[1e1, 1e4])
singular_values_plot(rom.model.to_python_control(), label="ROM", color="r", ls="--", omega=[1e1, 1e4])

plt.savefig(os.path.join(case_dir, "outputs", "sigmaplot.pdf"))

print('ok')