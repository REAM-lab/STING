# In progress - March 5, 2026 - Ruth 

# Import Python standard and third-party packages
from pathlib import Path
import os
from collections import namedtuple
import pylab as plt
import control as ct

# Import sting package
from sting import main
from sting.modules.model_order_reduction.reductions import SingularPerturbation, BalancedTruncation, IRKA
# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Target reduction order 
r = 8
zone_name = "zone_1"

# Open-loop modal reduction
reductions = {zone_name:  SingularPerturbation(r=r, basis="eigen")}
ssm, fom, open_mr = main.run_model_reduction(file_name="ssm_open_modal", case_directory=case_dir, reductions=reductions)
# Open-loop balanced reduction
gram = "subsystem"
method = "singular perturbation"
reductions = {zone_name:  BalancedTruncation(r=r, gramian_c=gram, gramian_o=gram, method=method)}
_, _, open_br = main.run_model_reduction(file_name="ssm_open_balanced", case_directory=case_dir, reductions=reductions)
# Closed-loop balanced reduction
gram = "lyapunov"
reductions = {zone_name:  BalancedTruncation(r=r, gramian_c=gram, gramian_o=gram, method=method)}
_, _, closed_br = main.run_model_reduction(file_name="ssm_closed_balanced", case_directory=case_dir, reductions=reductions)
# Open-loop IRKA
reductions = {zone_name:  IRKA(r=r)}
_, _, open_ki = main.run_model_reduction(file_name="ssm_open_irka", case_directory=case_dir, reductions=reductions)

# Save the full-order model
fom.to_csv(os.path.join(case_dir, "outputs", "ssm_full_order"))

# Plotting colors
red = "#BB5566"
yellow = "#DDAA33"
dark_blue = "#004488"
light_blue = "#6699CC"

# Define a named tuple class for a Point with 'x' and 'y' fields
Model = namedtuple('Model', ["name", "model", 'marker', 'color', 'line_style'])

models = [
    Model("Full-order model", fom, "x", "gray", "-"),
    Model("Modal Reduction", open_mr, "^", red, ":"),
    Model("Balanced Reduction (open)", open_br, "o", light_blue, "--"),
    Model("Balanced Reduction (closed)", closed_br, "o", dark_blue, "--"),
    Model("IRKA", open_ki, "s", yellow, "-.")
    ]

# Compare the eigenvalues of the FOM and ROMs
ax=plt.gca()
for m in models:
    facecolor = m.color if (m.marker == "x")  else "none"
    ax = m.model.plot_eigenvalues(ax=ax, marker=m.marker, label=m.name, edgecolor=m.color, facecolor=facecolor)
ax.set_xscale("symlog")
ax.legend()
plt.savefig(os.path.join(case_dir, "outputs", "eigenvalues.pdf"))
plt.close()

# Compare the sigmaplots of the FOM and ROMs
for m in models:
   out = ct.singular_values_plot(m.model.to_python_control(), omega=[1e1, 1e4], label=m.name, color=m.color, ls=m.line_style)
plt.ylim(1e-1, 1e2)
plt.savefig(os.path.join(case_dir, "outputs", "sigmaplot.pdf"))

print('ok')