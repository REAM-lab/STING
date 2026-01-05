# ----------------------
# Import python packages
# ----------------------
from __future__ import annotations
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
    system: System
    model: pyo.ConcreteModel = None
    solver_settings: dict = field(default_factory=lambda: {
                                                        "solver_name": "mosek_direct",
                                                        "tee": True,
                                                        })
    model_settings: dict = field(default_factory=lambda: {
                                                        "gen_costs": "quadratic",
                                                        "consider_shedding": False,
                                                        })
    output_directory: str = None
    
    def __post_init__(self):
        self.construct_model()
        self.set_output_folder()

    def set_output_folder(self):
        """
        Set up the output folder for storing results.
        """
        output_folder = os.path.join(self.system.case_directory, "outputs", "capacity_expansion")
        os.makedirs(output_folder, exist_ok=True)
        self.output_directory = output_folder

    def construct_model(self):
        """
        Construct the optimization model for capacity expansion.
        """
        
        # Create Pyomo model
        self.model = pyo.ConcreteModel()

        # Construct modules
        system = self.system
        model = self.model
        model_settings = self.model_settings
        generator.construct_capacity_expansion_model(system, model, model_settings)
        storage.construct_capacity_expansion_model(system, model, model_settings)
        bus.construct_capacity_expansion_model(system, model, model_settings)

        # Define objective function
        self.model.eCostPerTp = pyo.Expression(self.system.tp, expr=lambda m, t: m.eGenCostPerTp[t] + m.eStorCostPerTp[t])
        self.model.eCostPerPeriod = pyo.Expression(expr=lambda m: m.eGenCostPerPeriod + m.eStorCostPerPeriod)
        self.model.eTotalCost = pyo.Expression(expr=lambda m: sum(m.eCostPerTp[t] * t.weight for t in self.system.tp) + m.eCostPerPeriod)
        
        self.model.obj = pyo.Objective(expr=lambda m: m.eTotalCost, sense=pyo.minimize)

    def solve_model(self):

        solver = pyo.SolverFactory(self.solver_settings["solver_name"])
        results = solver.solve(self.model, tee=self.solver_settings["tee"])

        # Export results
        system, model, output_directory = self.system, self.model, self.output_directory

        bus.export_results_capacity_expansion(system, model, output_directory)
        print('ok')







