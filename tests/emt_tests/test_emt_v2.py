from sting.datasets.wscc_9 import wscc_9
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
from sting.modules.power_flow.utils import load_ac_power_flow_solution
from sting.utils.dynamical_systems import make_smooth_step

from numba import njit
import copy

import os 
# Set up a temporary directory used by all tests
case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)


sys = wscc_9(case_directory=case_directory)
sys.gfli_16a.clear()
sys.gfmi_18a.clear()

for b in ["bus_2", "bus_3", "bus_5"]:

    g = copy.deepcopy(sys.voltage_source_4a[0])
    g.bus = b
    sys.add(g)

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

from sting.modules.simulation_emt.core_v2 import SimulationEMT


emt_model = SimulationEMT.from_system(sys, jit=True)


input_signals = {
    "voltage_source_4a_0": {
        "v_ref_d": make_smooth_step(step_time=0.10, initial_value=0.0, final_value=0.10, transient_width=5e-3, jit=True),
    }
}

emt_model.simulate(t_max=1.5, input_signals=input_signals)

out_dir = os.path.join(case_directory, "outputs", "simulation_emt")
os.makedirs(out_dir, exist_ok=True)
emt_model.plot_results(output_directory=out_dir)

print("ok")