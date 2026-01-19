# ----------------------
# Import python packages
# ----------------------
from __future__ import annotations
from xml.parsers.expat import model
import polars as pl
from dataclasses import dataclass, field
import os
import pyomo.environ as pyo
import time
import logging
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output
import importlib

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

        logger.info("\n>> Starting capacity expansion...\n")
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
        logger.info(f"Solver settings: {self.solver_settings}")

        default = {
                "gen_costs": "quadratic",
                "consider_shedding": False,
                "consider_single_storage_injection": False,
                "consider_line_capacity": True,
                "consider_bus_max_flow": False,
                "consider_angle_limits": True,
                "policies": []
            }
        
        if self.model_settings is not None:
            for key, value in self.model_settings.items():
                default[key] = value
        
        self.model_settings = default
        logger.info(f"Model settings: {self.model_settings} \n")


    def set_output_folder(self):
        """
        Set up the output folder for storing results.
        """
        if self.output_directory is None:
            self.output_directory = os.path.join(self.system.case_directory, "outputs", "capacity_expansion")
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
        self.model.cost_components_per_period = []

        # Construct modules
        generator.construct_capacity_expansion_model(self.system, self.model, self.model_settings)
        storage.construct_capacity_expansion_model(self.system, self.model, self.model_settings)
        bus.construct_capacity_expansion_model(self.system, self.model, self.model_settings)

        if len(self.model_settings["policies"]) > 0:
            for policy in self.model_settings["policies"]:
                class_module = importlib.import_module(policy) 
                getattr(class_module, "construct_capacity_expansion_model")(self.system, self.model, self.model_settings)

        # Define objective function
        logger.info("> Initializing construction of objective function ...")
        start_time = time.time()

        def eCostPerTp_rule(m, t):
            return sum( getattr(m, tp_cost.name)[t] for tp_cost in m.cost_components_per_tp)
        
        def eCostPerPeriod_rule(m):
            return sum( getattr(m, period_cost.name) for period_cost in m.cost_components_per_period)

        self.model.eCostPerTp = pyo.Expression(self.system.tp, expr=eCostPerTp_rule)
        self.model.eCostPerPeriod = pyo.Expression(expr=eCostPerPeriod_rule)
        self.model.eTotalCost = pyo.Expression(expr= self.model.eCostPerPeriod + sum(self.model.eCostPerTp[t]  * t.weight for t in self.system.tp))
        
        self.model.rescaling_factor_obj = pyo.Param(initialize=1e-6)  # To express the objective in million USD

        self.model.obj = pyo.Objective(expr= self.model.rescaling_factor_obj * self.model.eTotalCost, sense=pyo.minimize)
        logger.info(f"> Completed in {time.time() - start_time:.2f} seconds. \n")



    def solve(self):
        """
        Solve the capacity expansion optimization model.
        """
        # Use root logger so solver output also goes to the file handler attached there
        start_time = time.time()
        logger.info("> Solving capacity expansion model...")
        solver = pyo.SolverFactory(self.solver_settings["solver_name"])
        
        # Write solver output to sting_log.txt
        with capture_output(output=LogStream(logger=logging.getLogger(), level=logging.INFO)):
            results = solver.solve(self.model, options=self.solver_settings['solver_options'], tee=self.solver_settings['tee'])

        # Load the duals into the 'dual' suffix
        solver.load_duals()

        logger.info(f"> Time spent by solver: {time.time() - start_time:.2f} seconds.")
        logger.info(f"> Solver finished with status: {results.solver.status}, termination condition: {results.solver.termination_condition}.")
        logger.info(f"> Objective value: {(pyo.value(self.model.obj) * 1/self.model.rescaling_factor_obj):.2f} USD.")

        with open(os.path.join(self.output_directory, 'model_output.txt'), 'w') as output_file:
            self.model.pprint(ostream=output_file)

        self.export_results_to_csv()

    @timeit
    def export_results_to_csv(self):
        """
        Export all results to CSV files.
        """
        logger.info(f"- Directory: {self.output_directory}")

        # Export costs summary
        costs = pl.DataFrame({'component' : ['CostPerTimepoint_USD', 'CostPerPeriod_USD', 'TotalCost_USD'],
                              'cost' : [  sum( pyo.value(self.model.eCostPerTp[t]) * t.weight for t in self.system.tp), 
                                            pyo.value(self.model.eCostPerPeriod), 
                                            pyo.value(self.model.eTotalCost)]})
        costs.write_csv(os.path.join(self.output_directory, 'costs_summary.csv'))

        generator.export_results_capacity_expansion(self.system, self.model, self.output_directory)
        storage.export_results_capacity_expansion(self.system, self.model, self.output_directory)
        bus.export_results_capacity_expansion(self.system, self.model, self.output_directory)

        if len(self.model_settings["policies"]) > 0:
            for policy in self.model_settings["policies"]:
                class_module = importlib.import_module(policy) 
                getattr(class_module, "export_results_capacity_expansion")(self.system, self.model, self.output_directory)







