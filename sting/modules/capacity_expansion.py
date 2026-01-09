# ----------------------
# Import python packages
# ----------------------
from __future__ import annotations
from pyexpat import model
import polars as pl
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
    solver_settings: dict = None
    model_settings: dict = None
    output_directory: str = None
    
    def __post_init__(self):

        self.set_settings()
        self.construct()
        self.set_output_folder()

    def set_settings(self):
        default =  {
                "solver_name": "gurobi",
                "tee": True,
                "solver_options": {},
            }
        if self.solver_settings is not None:
            for key, value in self.solver_settings.items():
                default[key] = value
        
        self.solver_settings = default

        default = {
                "gen_costs": "quadratic",
                "consider_shedding": False,
                "consider_single_storage_injection": False,
            }
        
        if self.model_settings is not None:
            for key, value in self.model_settings.items():
                default[key] = value
        
        self.model_settings = default

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
        self.model.eCostPerTp = pyo.Expression(self.system.tp, expr=lambda m, t: m.eGenCostPerTp[t] + m.eStorCostPerTp[t] + (m.eShedCostPerTp[t] if model_settings["consider_shedding"] else 0))
        self.model.eCostPerPeriod = pyo.Expression(expr=lambda m: m.eGenCostPerPeriod + m.eStorCostPerPeriod + m.eLineCostPerPeriod)
        self.model.eTotalCost = pyo.Expression(expr= (sum(self.model.eCostPerTp[t] * t.weight for t in self.system.tp) + self.model.eCostPerPeriod))
        
        self.model.rescaling_factor_obj = pyo.Param(initialize=1e-6)  # To express the objective in million USD

        self.model.obj = pyo.Objective(expr= self.model.rescaling_factor_obj * self.model.eTotalCost, sense=pyo.minimize)
        print(f"ok [{time.time() - start_time:.2f} seconds].")

        full_end_time = time.time()
        print(f"        Total: {full_end_time - full_start_time:.2f} seconds.")

    def solve(self):
        """
        Solve the capacity expansion optimization model.
        """

        start_time = time.time()
        print("> Solving capacity expansion model...")
        solver = pyo.SolverFactory(self.solver_settings["solver_name"])
        results = solver.solve(self.model, options=self.solver_settings['solver_options'], tee=self.solver_settings["tee"])

        # Load the duals into the 'dual' suffix
        solver.load_duals()

        print(f"> Time spent by solver: {time.time() - start_time:.2f} seconds.")
        print(f"> Solver finished with status: {results.solver.status}, termination condition: {results.solver.termination_condition}.")
        print(f"> Objective value: {(pyo.value(self.model.obj) * 1/self.model.rescaling_factor_obj):.2f} USD.")

        # Export costs summary
        costs = pl.DataFrame({'component' : ['CostPerTimepoint_USD', 'CostPerPeriod_USD', 'TotalCost_USD'],
                              'cost' : [  sum( pyo.value(self.model.eCostPerTp[t]) * t.weight for t in self.system.tp), 
                                            pyo.value(self.model.eCostPerPeriod), 
                                            pyo.value(self.model.eTotalCost)]})
        costs.write_csv(os.path.join(self.output_directory, 'costs_summary.csv'))

        start_full_time = time.time()
        system, model, output_directory = self.system, self.model, self.output_directory
        print(f"> Exporting results in {output_directory} :")
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







