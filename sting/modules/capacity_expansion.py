# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
import itertools
from more_itertools import transpose
from typing import NamedTuple, Optional, ClassVar
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyomo.environ as pyo

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.timescales.core import timescale_calculations
import sting.system.selections as sl
from sting.utils.dynamical_systems import DynamicalVariables
import sting.bus.bus as bus
import sting.timescales.core as timescales
import sting.generator.generator as generator
import sting.generator.storage as storage

# -----------
# Main class
# -----------
@dataclass(slots=True)
class CapacityExpansion:
    """
    Class to perform capacity expansion analysis of a power system.

    #### Attributes:
    - system: `System`
            The system to be analyzed.
    - planning_horizon: `int`
            The planning horizon in years.
    - investment_options: `list`
            List of investment options available.
    - results: `dict`
            Dictionary to store the results of the analysis.
    """
    system: System 
    model: pyo.ConcreteModel = field(init=False, default=None)
    model_settings: dict = field(default_factory=lambda: {
                                                        "gen_costs": "quadratic",
                                                        "consider_shedding": False,
                                                        })

    def __post_init__(self):
        self.system.clean_up()
        self.extra_calculations()
        self.construct_model()

    def extra_calculations(self):
        """
        Perform the capacity expansion analysis.
        """
        timescale_calculations(self.system.tp, self.system.ts)
        
        for load in self.system.load:
            load.assign_indices(self.system.bus, self.system.sc, self.system.tp)

        for line in self.system.line:
            line.assign_indices(self.system.bus)

    def construct_model(self):
        """
        Construct the optimization model for capacity expansion.
        """
        self.model = pyo.ConcreteModel()

        timescales.construct_capacity_expansion_model(self.system, self.model, self.model_settings)
        generator.construct_capacity_expansion_model(self.system, self.model, self.model_settings)
        storage.construct_capacity_expansion_model(self.system, self.model, self.model_settings)
        bus.construct_capacity_expansion_model(self.system, self.model, self.model_settings)

        self.model.eCostPerTp = pyo.Expression(self.system.tp, expr=lambda m, t: m.eGenCostPerTp[t] + m.eStorCostPerTp[t])
        self.model.eCostPerPeriod = pyo.Expression(expr=lambda m: m.eGenCostPerPeriod + m.eStorCostPerPeriod)
        self.model.eTotalCost = pyo.Expression(expr=lambda m: sum(m.eCostPerTp[t] * t.weight for t in self.system.tp) + m.eCostPerPeriod)
        
        self.model.obj = pyo.Objective(expr=lambda m: m.eTotalCost, sense=pyo.minimize)

        solver = pyo.SolverFactory('mosek_direct')
        results = solver.solve(self.model, tee=True)


        print('ok')
        # Define sets, parameters, variables, constraints, and objective function here






