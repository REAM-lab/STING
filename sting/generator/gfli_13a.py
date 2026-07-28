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
from sting.components import PhaseLockedLoop2A, InnerCurrentController2A, LCLFilter6A
