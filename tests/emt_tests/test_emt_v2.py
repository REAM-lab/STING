from sting.datasets.wscc_9 import wscc_9
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
from sting.modules.power_flow.utils import load_ac_power_flow_solution
from sting.utils.dynamical_systems import smooth_step_jit
from sting.load.core import Load




from numba import njit
import copy

import os 
# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)



import os

import polars as pl

from sting import datasets, main
from sting.generator import GFLI16A
from sting.utils.plotting_tools import compare_timeseries

# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# -------------------------------------------------------
# Construct a simple 2-bus system
# -------------------------------------------------------
gfli_1 = GFLI16A(
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
    kp_pc_pu=0.1, ki_pc_puHz=100
)

sys = datasets.toy_2(case_directory=case_directory)
sys.add(gfli_1)

load_1 = Load(bus="bus_1", zone="external", timepoint="t1", load_MW=0, load_MVAR=0)
sys.loads.clear()
sys.add(load_1)

sys.apply("post_system_init", sys)





# Run power flow
pf = ACPowerFlow(system=sys)
pf.solve()

# Break down lines into branches and shunts for small-signal modeling
sys_modifier = SystemModifier(system=sys)
sys_modifier.decompose_lines()
sys_modifier.combine_shunts()
sys_modifier.create_impedance_loads()

pf_sol = load_ac_power_flow_solution(pf.output_directory)

t = sys.timepoints[0]

sys.apply("load_ac_power_flow_solution", t.name, pf_sol)

from sting.modules.simulation_emt.core_v3 import SimulationEMT


emt_model = SimulationEMT.from_system(sys)

import numpy as np
from numba import njit

@njit
def inputs(t, x):
    u = np.zeros(4)
    u[2] = smooth_step_jit(t, step_time=0.10, initial_value=0.0, final_value=0.10, transient_width=5e-3)
    return u

emt_model.simulate(t_max=1.5, inputs=inputs)

out_dir = os.path.join(case_directory, "outputs", "simulation_emt")
os.makedirs(out_dir, exist_ok=True)
emt_model.plot_results(output_directory=out_dir)

print("ok")