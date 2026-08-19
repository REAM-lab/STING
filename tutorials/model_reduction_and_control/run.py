import os
from pathlib import Path

import control as ct
import numpy as np
from control_design import construct_controller
from wscc_9 import wscc_9

from sting import main
from sting.modules.model_order_reduction.balanced_truncation import BalancedTruncation
from sting.utils.dynamical_systems import make_smooth_step
from sting.utils.transformations import abc2dq0

# ------------------------------------------------------------
# Setup output file paths
# ------------------------------------------------------------
# Current working directory
cwd = os.path.join(Path(__file__).resolve().parent)

dir_ssm = os.path.join(cwd, "outputs", "small_signal_model")
dir_rom = os.path.join(cwd, "outputs", "model_order_reduction")
dir_with_ctr = os.path.join(cwd, "outputs", "emt_with_control")
dir_without_ctr = os.path.join(cwd, "outputs", "emt_without_control")

# ------------------------------------------------------------
# Construct the WSCC 9 bus system and simulation setup
# ------------------------------------------------------------
system = wscc_9(case_directory=cwd)

# Create input signal to the proposed inverter project
step = make_smooth_step(step_time=0.1, initial_value=0.0, final_value=0.10, transient_width=5e-3)
inputs = {'gfmi_18a_0': {'v_ref': step}}
t_max = 1.5 # Simulation length in seconds

# ------------------------------------------------------------
# Run an EMT simulation 
# ------------------------------------------------------------
#main.run_emt(system=system, inputs=inputs, t_max=t_max, output_directory=dir_without_ctr)

# ------------------------------------------------------------
# Construct a small-signal model (SSM)
# ------------------------------------------------------------
system, ssm = main.run_ssm(system=system)
# Simulate the full-order model
ssm.simulate_ssm(inputs=inputs, t_max=t_max, output_directory=dir_ssm)

# ------------------------------------------------------------
# Construct a reduced-order model (ROM)
# ------------------------------------------------------------
balanced_truncation = {
    "external": BalancedTruncation(r=33, method="truncate")
    }
rom = main.run_model_reduction(ssm=ssm, reductions=balanced_truncation)
# Simulate the reduced-order model
rom.simulate_ssm(inputs=inputs, t_max=t_max, output_directory=dir_rom)

# Compute statistics of the ROM and FOM
external_grid = rom.system.linear_subsystems[0]
ss_fom = ct.ss(*external_grid.full_order_model.data)
ss_rom = ct.ss(*external_grid.reduced_order_model.data)
print("Full-order model has", ss_fom.nstates, "states")
print("Reduced-order model (without proposed project) ", ss_rom.nstates, "states")
print("H_2 Error", round(100 * ct.norm(ss_fom - ss_rom,p=2) / ct.norm(ss_fom, p=2),3), "%")
print("Max eigenvalue of the ROM + study area: ", np.max(np.linalg.eigvals(rom.model.A).real))

# ------------------------------------------------------------
# Output feedback control design
# ------------------------------------------------------------
F = construct_controller(rom)

# Initial conditions in the LCL filter
w0 = 1
x0 = rom.system.gfmi_18a[0].lcl_filter.emt_init
y0 = np.array([w0, x0.i_vsc_d, x0.i_vsc_q, x0.i_bus_d, x0.i_bus_q])

def output_feedback_control(t: float, x: np.ndarray, id: dict):
    # Unpack the states of the GFM
    i_vsc_abc = (x[id['gfmi_18a_0']['i_vsc_'+p]] for p in ['a','b','c'])
    i_bus_abc = (x[id['gfmi_18a_0']['i_bus_'+p]] for p in ['a','b','c'])
    angle = x[id['gfmi_18a_0']['angle']]
    w = x[id['gfmi_18a_0']['w']]

    # Transform abc to dq0  
    i_vsc_d, i_vsc_q, _ = abc2dq0(*i_vsc_abc, angle)
    i_bus_d, i_bus_q, _ = abc2dq0(*i_bus_abc, angle)

    # Control action
    y = np.array([w, i_vsc_d, i_vsc_q, i_bus_d, i_bus_q])
    delta_u = F @ (y - y0)

    return delta_u[0]

controller = {'gfmi_18a_0': {'v_ref': step, 'p_ref': output_feedback_control}}

# ------------------------------------------------------------
# Simulate the EMT after controller placement
# ------------------------------------------------------------
main.run_emt(system=system, inputs=controller, t_max=t_max, output_directory=dir_with_ctr)