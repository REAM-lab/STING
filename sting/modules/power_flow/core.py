# ----------------------
# Import python packages
# ----------------------
import polars as pl
from dataclasses import dataclass
import os
import pyomo.environ as pyo
import time
import logging
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output
from pyomo.environ import *


# ------------------
# Import sting code
# ------------------
from sting.system.core_testing import System
from sting.modules.power_flow.utils import ModelSettings, SolverSettings, ACPowerFlowSolution
import sting.bus.ac_power_flow as bus
import sting.generator.generator as generator
from sting.utils.data_tools import timeit

logger = logging.getLogger(__name__)


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
    solution: ACPowerFlowSolution = None

    def __post_init__(self):
        logger.info("\n>> Starting AC power flow...\n")
        self.set_settings()
        self.set_output_folder()
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
        bus.construct_ac_power_flow_model(self)

        # Construct objective function
        start_time = time.time()

        def eCostPerTp_rule(m, t):
            return sum( getattr(m, tp_cost.name)[t] for tp_cost in m.cost_components_per_tp)
        
        self.model.eCostPerTp = pyo.Expression(self.system.timepoints, expr=eCostPerTp_rule)
        self.model.eTotalCost = pyo.Expression(expr= sum(self.model.eCostPerTp[t]  * t.weight for t in self.system.timepoints))
        
        self.model.rescaling_factor_obj = pyo.Param(initialize=1) 

        self.model.obj = pyo.Objective(expr= self.model.rescaling_factor_obj * self.model.eTotalCost, sense=pyo.minimize)
        logger.info(f"> Completed in {time.time() - start_time:.2f} seconds. \n")

    def solve(self):
        """
        Solve the ac power flow optimization model.
        """
        # Use root logger so solver output also goes to the file handler attached there
        start_time = time.time()
        logger.info("> Solving ac power flow model...")
        solver = pyo.SolverFactory(self.solver_settings.solver_name)
        
        # Write solver output to sting_log.txt
        with capture_output(output=LogStream(logger=logging.getLogger(), level=logging.INFO)):
            if self.solver_settings.solver_options is not None:
                results = solver.solve(self.model, options=self.solver_settings.solver_options, tee=self.solver_settings.tee)
            else:
                results = solver.solve(self.model, tee=self.solver_settings.tee)

        # Load the duals into the 'dual' suffix
        try:
            solver.load_duals()
        except:
            logger.warning("Could not load duals, i.e., shadow prices, from solver.")

        logger.info(f"> Time spent by solver: {time.time() - start_time:.2f} seconds.")
        logger.info(f"> Solver finished with status: {results.solver.status}, termination condition: {results.solver.termination_condition}.")
        logger.info(f"> Objective value: {(pyo.value(self.model.obj))}. \n")

        self.model.solver_status = results.solver.status
        self.model.termination_condition = results.solver.termination_condition
        self.model.solver_time_spent = str(time.time() - start_time)

        if self.model_settings.write_model_file:
            with open(os.path.join(self.output_directory, 'model_output.txt'), 'w') as output_file:
                self.model.pprint(ostream=output_file)

        self.export_results_to_csv()


    @timeit
    def export_results_to_csv(self):
        """
        Export all results to CSV files.
        """
        logger.info(f"- Directory: {self.output_directory}")

        # Export solver results summary
        solver_status = pl.DataFrame({'attribute' : ['solver_name', 'solver_status', 'termination_condition', 'time_spent_seconds'],
                                      'value' : [ self.solver_settings.solver_name, 
                                                  self.model.solver_status,
                                                  self.model.termination_condition,
                                                  self.model.solver_time_spent]})
        solver_status.write_csv(os.path.join(self.output_directory, 'solver_status.csv'))

        # Export costs summary
        costs = pl.DataFrame({'component' : ['total_cost_USD'],
                              'cost' : [  pyo.value(self.model.eTotalCost)]})
        costs.write_csv(os.path.join(self.output_directory, 'costs_summary.csv'))

        generator.export_results_ac_power_flow(self)
        bus.export_results_ac_power_flow(self)
        



