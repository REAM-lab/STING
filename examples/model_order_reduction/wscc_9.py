import os
from collections import namedtuple
from pathlib import Path

import pylab as plt

# Import sting package
from sting import main
from sting.system import System
from sting.modules.model_order_reduction.reductions import (
    IRKA,
    BalancedTruncation,
    SingularPerturbation,
)
from sting.utils.dynamical_systems import smooth_step

case_dir = os.path.join(Path(__file__).resolve().parent, "wscc_9")

### Load Data ###
# Western System Coordinating Council (WSCC) 9 bus system
wscc_9 = System.from_dataset("wscc_9")
wscc_9.case_directory = case_dir
# Construct a small-signal model
wscc_9, ssm = main.run_ssm(system=wscc_9)


### Model reductions ###
r = 5                  # Target reduction order 
zone_name = "external"  # Zone to reduce

# Settings for each model reduction method
# Eliminate the fastest modes from the state-space model
singular_perturbation = {
    zone_name:  SingularPerturbation(r=r, basis="eigen")
    }
# Vanilla balanced truncation removing the states that are hardest to control and observe
balanced_truncation = {
    zone_name:  BalancedTruncation(r=r, gramian_c="lyapunov", gramian_o="lyapunov", method="truncate")
    }
# Balanced truncation variant for interconnected subsystems
subsystem_balanced_truncation = {
    zone_name:  BalancedTruncation(r=r, gramian_c="subsystem", gramian_o="subsystem", method="singular perturbation")
    }

ssm_sp = main.run_model_reduction(ssm=ssm, reductions=singular_perturbation)
ssm_bt = main.run_model_reduction(ssm=ssm, reductions=balanced_truncation)
ssm_sbt = main.run_model_reduction(ssm=ssm, reductions=subsystem_balanced_truncation)

## Simulate the response of the GFMI with each reduced-order model ##
inputs = {
    'gfmi_18a_0': {
        'p_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=0.10, transient_width=5e-3),
        'q_ref': lambda t: smooth_step(t, step_time=0.10, initial_value=0.0, final_value=-0.10, transient_width=5e-3)
        }
}

for model, dir in zip([ssm, ssm_sp, ssm_bt, ssm_sbt], ["full_order_model", "singular_perturbation", "balanced_truncation", "subsystem_balanced_truncation"]):
    t_max = 1.5 # Simulation length in seconds
    model.output_directory = os.path.join(case_dir, dir)
    os.makedirs(model.output_directory , exist_ok=True)
    model.simulate_ssm(t_max=t_max, inputs=inputs)


### Plot eigenvalues ###
# Plotting colors
red = "#BB5566"
yellow = "#DDAA33"
dark_blue = "#004488"
light_blue = "#6699CC"

# Define a named tuple class for a Point with 'x' and 'y' fields
Model = namedtuple('Model', ["name", "model", 'marker', 'color', 'line_style'])

models = [
    Model("Full-order model", ssm, "x", "gray", "-"),
    Model("Modal Reduction", ssm_sp, "^", red, ":"),
    Model("Balanced Reduction (open)", ssm_bt, "s", light_blue, "--"),
    Model("Balanced Reduction (closed)", ssm_sbt, "o", dark_blue, "--"),
    ]

# Compare the eigenvalues of the FOM and ROMs
ax=plt.gca()
for m in models:
    facecolor = m.color if (m.marker == "x")  else "none"
    ax = m.model.model.plot_eigenvalues(ax=ax, marker=m.marker, label=m.name, edgecolor=m.color, facecolor=facecolor)
ax.set_xscale("symlog")
ax.legend()
plt.savefig(os.path.join(case_dir, "outputs", "eigenvalues.pdf"))
plt.close()

