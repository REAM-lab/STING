# ----------------------
# Import python packages
# ----------------------
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
import os
import pyomo.environ as pyo
import time

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
import sting.system.selections as sl
import sting.bus.bus as bus
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
        self.construct()
        self.set_output_folder()

    def set_output_folder(self):
        """
        Set up the output folder for storing results.
        """
        output_folder = os.path.join(self.system.case_directory, "outputs", "capacity_expansion")
        os.makedirs(output_folder, exist_ok=True)
        self.output_directory = output_folder

    def construct(self):
        """
        Construct the optimization model for capacity expansion.
        """
        
        print("> Constructing capacity expansion model:")
        full_start_time = time.time()

        # Create Pyomo model
        self.model = pyo.ConcreteModel()

        # Construct modules
        system = self.system
        model = self.model
        model_settings = self.model_settings

        print("   - Generators variables and constraints ...", end=' ')
        start_time = time.time()
        generator.construct_capacity_expansion_model(system, model, model_settings)
        print(f"ok [{time.time() - start_time:.2f} seconds].")

        print("   - Storage variables and constraints ...", end=' ')
        start_time = time.time()
        storage.construct_capacity_expansion_model(system, model, model_settings)
        print(f"ok [{time.time() - start_time:.2f} seconds].")
        
        print("   - Bus variables and constraints ...", end=' ')
        start_time = time.time()
        bus.construct_capacity_expansion_model(system, model, model_settings)
        print(f"ok [{time.time() - start_time:.2f} seconds].")

        # Define objective function
        print("   - Objective function ...", end=' ')
        start_time = time.time()
        self.model.eCostPerTp = pyo.Expression(self.system.tp, expr=lambda m, t: m.eGenCostPerTp[t] + m.eStorCostPerTp[t])
        self.model.eCostPerPeriod = pyo.Expression(expr=lambda m: m.eGenCostPerPeriod + m.eStorCostPerPeriod)
        self.model.eTotalCost = pyo.Expression(expr=lambda m: sum(m.eCostPerTp[t] * t.weight for t in self.system.tp) + m.eCostPerPeriod)
        self.model.obj = pyo.Objective(expr=lambda m: m.eTotalCost, sense=pyo.minimize)
        print(f"ok [{time.time() - start_time:.2f} seconds].")

        full_end_time = time.time()
        print(f"        Total: {full_end_time - full_start_time:.2f} seconds.")

    def solve(self):
        """
        Solve the capacity expansion optimization model.
        """

        print("> Solving capacity expansion model...")
        solver = pyo.SolverFactory(self.solver_settings["solver_name"])
        results = solver.solve(self.model, tee=self.solver_settings["tee"])

        print(f"> Solver finished with status: {results.solver.status}, termination condition: {results.solver.termination_condition}.")
        print(f"> Objective value: {pyo.value(self.model.obj):.2f} USD.")

        print("> Exporting capacity expansion results:")
        start_full_time = time.time()
        system, model, output_directory = self.system, self.model, self.output_directory

        print("   - Generators results ...", end=' ')
        start_time = time.time()
        generator.export_results_capacity_expansion(system, model, output_directory)
        print(f"ok [{time.time() - start_time:.2f} seconds].")

        print("   - Storage results ...", end=' ')
        start_time = time.time()
        storage.export_results_capacity_expansion(system, model, output_directory)
        print(f"ok [{time.time() - start_time:.2f} seconds].")

        print("   - Bus results ...", end=' ')
        start_time = time.time()
        bus.export_results_capacity_expansion(system, model, output_directory)
        print(f"ok [{time.time() - start_time:.2f} seconds].")

        full_end_time = time.time()
        print(f"        Total: {full_end_time - start_full_time:.2f} seconds.")







