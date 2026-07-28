"""
This module implements a 13th order Grid-following Inverter comprised of: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Current controller: A dq-based frame PI controller
- PLL with filter: It that tracks the phase of the grid voltage.
- Reactive power controller: A PI controller that regulates the reactive power of the inverter.
- Active power controller: A PI controller that regulates the active power of the inverter.
"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field

# ------------------
# Import sting code
# ------------------
from sting.generator.core import Generator
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.transformations import dq02abc, abc2dq0
from sting.components import PhaseLockedLoop2A, InnerCurrentController2A, LCLFilter6A, ActivePowerPI1A, ReactivePowerPI1A


@dataclass(slots=True, kw_only=True, eq=False)
class GFLI10A(Generator):
    # LCL filter parameters
    rf1_pu: float
    xf1_pu: float
    csh_pu: float
    rsh_pu: float
    txr_power_MVA: float
    txr_voltage1_kV: float
    txr_voltage2_kV: float
    txr_r1_pu: float
    txr_x1_pu: float
    txr_r2_pu: float
    txr_x2_pu: float
    # Phase-locked loop parameters
    kp_pll_pu: float
    ki_pll_puHz: float
    # Current controller parameters
    kp_cc_pu: float
    ki_cc_puHz: float
    kff_cc: float

    # Components
    lcl_filter: LCLFilter6A = field(init=False)
    current_controller: InnerCurrentController2A = field(init=False)
    phase_locked_loop: PhaseLockedLoop2A = field(init=False)

    def __post_init__(self):
        self.lcl_filter = LCLFilter6A(self.rf1_pu, self.xf1_pu, self.rsh_pu, self.csh_pu, self.rf2_pu, self.xf2_pu, self.wbase)
        self.phase_locked_loop = PhaseLockedLoop2A(self.kp_pll_pu, self.ki_pll_puHz, self.wbase)
        self.current_controller = InnerCurrentController2A(self.kp_cc_pu, self.ki_cc_puHz, self.kff_cc, self.xf1_pu + self.xf2_pu)