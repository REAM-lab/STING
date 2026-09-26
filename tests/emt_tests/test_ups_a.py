import os
import numpy as np

from sting import main
from sting.system import System

# Core components
from sting.generator import UPSA, VoltageSource4A
from sting.bus import Bus
from sting.line import LinePiModel
from sting.load import Load
from sting.timescales import Timepoint
from sting.modules.simulation_emt.core import SimulationEMT


case_directory = os.path.join(os.getcwd(), "tests", "emt_tests", "tmpdir")
os.makedirs(case_directory, exist_ok=True)

# Construct the paper's single-data-center infinite-bus (SDCIB) system.
t1 = Timepoint(name="t1", weight=1)
bus_1 = Bus(
    name="infinite_bus",
    bus_type="slack",
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    minimum_voltage_pu=1,
    maximum_voltage_pu=1,
)
bus_2 = Bus(
    name="pcc",
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    minimum_voltage_pu=0.95,
    maximum_voltage_pu=1.3,
)
load_1 = Load(bus="pcc", timepoint="t1", load_MW=0, load_MVAR=0)
line_1 = LinePiModel(
    name="infinite_source_impedance",
    from_bus="infinite_bus",
    to_bus="pcc",
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    r_pu=0.02,
    x_pu=0.19,
    g_pu=0.0005,
    b_pu=0.001,
)
source = VoltageSource4A(
    name="infinite_source",
    bus="infinite_bus",
    slack=True,
    minimum_active_power_MW=-200,
    maximum_active_power_MW=200,
    minimum_reactive_power_MVAR=-500,
    maximum_reactive_power_MVAR=500,
    cost_variable_USDperMWh=0,
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
    r_pu=0.001,
    x_pu=0.005,
)
ups = UPSA(
    name="santiago_ups",
    bus="pcc",
    minimum_active_power_MW=-50,
    maximum_active_power_MW=-50,
    minimum_reactive_power_MVAR=0,
    maximum_reactive_power_MVAR=0,
    cost_variable_USDperMWh=10,
    base_power_MVA=100,
    base_voltage_kV=230,
    base_frequency_Hz=60,
)

system = System(case_directory=case_directory)
for component in [bus_1, bus_2, load_1, line_1, source, ups, t1]:
    system.add(component)
system.apply("post_system_init", system)


def load_step(t):
    return 0.05 if t >= 0.1 else 0.0


inputs = {}


_, ssm = main.run_ssm(system=system, case_directory=case_directory)
#ups.define_variables_emt()
#assert list(ups.ssm.x.name) == ups._state_names()
#assert (ups.ssm.x.init == ups.variables_emt.x.init).all()
#assert np.all(np.isfinite(ssm.model.x.init))
#assert np.max(np.real(np.linalg.eigvals(ssm.model.A))) < 0.0

emt = SimulationEMT(system=system)
assert np.all(np.isfinite(emt.variables.x.init))
x0 = emt.variables.x.init
u_device = emt.build_device_input({}, x0, 0.0)
y0 = emt.build_stacked_output(x0)
F_abc, G_abc, _, _ = emt.ccm_abc_matrices
u_stack = F_abc @ y0 + G_abc @ u_device
dx0 = emt.build_state_derivative(x0, u_stack)
assert np.all(np.isfinite(dx0))
for component_name, state_indices in emt.x_idx.items():
    print(component_name, "initial max derivative", np.max(np.abs(dx0[state_indices])))

ssm.simulate_ssm(t_max=1.5, inputs=inputs)

main.run_emt(
    inputs=inputs,
    t_max=1.5,
    system=system,
    case_directory=case_directory,
)