import os

from sting import main, datasets
from sting.system import System

# Core components
from sting.generator import VoltageSource4A, GFLI16A
from sting.line import LinePiModel
from sting.bus import Bus
from sting.load import Load
from sting.timescales import Timepoint

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
    kp_pll_pu=100, ki_pll_puHz=2500, tau_pll=1/100,
    # Inner current controller
    kp_cc_pu=0.05, ki_cc_puHz=0.6, kff_cc=0.75,
    # Power controllers
    kp_pc_pu=0.1, ki_pc_puHz=100
)

system = datasets.toy_2(case_directory=case_directory)
system.add(gfli_1)
system.apply("post_system_init", system)

# -------------------------------------------------------
# Run small-signal model and EMT simulations
# -------------------------------------------------------

# Step function inputs to simulate
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return -0.1 if t >= 0.5 else 0.0

inputs = {
    'gfli_16a_0': {
        'p_ref': step1,
        'q_ref': step2,
        }
}

t_max = 1.5 # Simulation length in seconds


# Construct system and small-signal model
_, ssm = main.run_ssm(system=system, case_directory=case_directory)
ssm.simulate_ssm(t_max=t_max, inputs=inputs)
# Run EMT simulation
main.run_emt(inputs=inputs, t_max=t_max, system=system, case_directory=case_directory)