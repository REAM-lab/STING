# ----------------------
# Import python packages
# ----------------------
from __future__ import annotations
import polars as pl
import numpy as np
from dataclasses import dataclass, field
import os
import pyomo.environ as pyo
import time
import logging
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output
import importlib
from typing import NamedTuple

from pyomo.environ import *
from pyomo.repn import generate_standard_repn
import math

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
import sting.system.selections as sl
import sting.bus.bus as bus
import sting.generator.generator as generator
import sting.generator.storage as storage
from sting.utils.data_tools import timeit

logger = logging.getLogger(__name__)

# -----------
# Sub-classes 
# -----------
class ModelSettings(NamedTuple):
     pass

class SolverSettings(NamedTuple):
    """
    Settings for the solver for the capacity expansion model.
    """
    solver_name: str = "mosek_direct"
    tee: bool = True
    solver_options: dict = field(default_factory=dict)

# -----------
# Main class
# -----------
@dataclass(slots=True)
class ACPowerFlow:
    """
    Class for AC power flow model.
    """
    system: System
    model: pyo.ConcreteModel = None
    model_settings: ModelSettings = None
    solver_settings: SolverSettings = None
    output_directory: str = None

    def __post_init__(self):
        logger.info("\n>> Starting AC power flow...\n")
        self.set_settings()
        self.construct()

    def set_settings(self):

        if self.model_settings is None:
            self.model_settings = ModelSettings()
        else:
            self.model_settings = ModelSettings(**self.model_settings)

        logger.info(f"Model settings: {self.model_settings}")

        if self.solver_settings is None:
            self.solver_settings = SolverSettings()
        else:
            self.solver_settings = SolverSettings(**self.solver_settings)

        logger.info(f"Solver settings: {self.solver_settings}")

    def set_output_folder(self):
        """
        Set up the output folder for storing results.
        """
        if self.output_directory is None:
            self.output_directory = os.path.join(self.system.case_directory, "outputs", "ac_power_flow")
        os.makedirs(self.output_directory, exist_ok=True)

    @timeit    
    def construct(self):
        """
        Construction of the optimization model for capacity expansion.
        """
        
        # Create Pyomo model
        self.model = pyo.ConcreteModel()

        # Construct empty lists for costs
        self.model.cost_components_per_tp = []

        # Construct modules
        generator.construct_ac_power_flow_model(self)

        """
        storage.construct_ac_power_flow_model(self)
        bus.construct_ac_power_flow_model(self)

        # Define objective function
        logger.info("> Initializing construction of objective function ...")
        start_time = time.time()

        def eCostPerTp_rule(m, t):
            return sum( getattr(m, tp_cost.name)[t] for tp_cost in m.cost_components_per_tp)
        
        self.model.eCostPerTp = pyo.Expression(self.system.tp, expr=eCostPerTp_rule)
        self.model.eTotalCost = pyo.Expression(expr= sum(self.model.eCostPerTp[t]  * t.weight for t in self.system.tp))
        
        self.model.rescaling_factor_obj = pyo.Param(initialize=1) 

        self.model.obj = pyo.Objective(expr= self.model.rescaling_factor_obj * self.model.eTotalCost, sense=pyo.minimize)
        logger.info(f"> Completed in {time.time() - start_time:.2f} seconds. \n")
        """