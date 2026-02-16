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
from sting.system.core_testing import System
import sting.system.selections as sl
import sting.bus.core as core
import sting.generator.generator as generator
import sting.generator.storage as storage
from sting.utils.data_tools import timeit

logger = logging.getLogger(__name__)

# -----------
# Sub-classes 
# -----------
class ModelSettings(NamedTuple):   
    """
    Settings for the capacity expansion model.
    """
    generator_type_costs: str = "linear"
    load_shedding: bool = False
    single_storage_injection: bool = False
    generation_capacity_expansion: bool = True
    storage_capacity_expansion: bool = True
    line_capacity_expansion: bool = True
    line_capacity: bool = True
    power_flow: str = "dc"
    bus_max_flow_expansion: bool = False
    bus_max_flow: bool = False
    angle_difference_limits: bool = False
    policies: list[int] = None
    write_model_file: bool = False
    kron_equivalent_flow_constraints: bool = False

class SolverSettings(NamedTuple):
    """
    Settings for the solver for the capacity expansion model.
    """
    solver_name: str = "mosek_direct"
    tee: bool = True
    solver_options: dict = field(default_factory=dict)

class KronVariables(NamedTuple):
    original_system: System
    removable_buses: set[str] = None
    Y_original: np.ndarray = None
    Y_kron: np.ndarray = None
    B_qp: np.ndarray = None
    invB_qq: np.ndarray = None

# -----------
# Main class
# -----------
@dataclass(slots=True)
class CapacityExpansion:
    system: System
    model: pyo.ConcreteModel = None
    model_settings: ModelSettings = None
    solver_settings: SolverSettings = None
    output_directory: str = None
    kron_variables: KronVariables = None
    
    def __post_init__(self):

        logger.info("\n>> Starting capacity expansion...\n")
        self.set_settings()
        self.construct()
        self.inspect_coefficients()
        self.set_output_folder()

    def set_settings(self):

        if self.model_settings is None:
            self.model_settings = ModelSettings()
        else:
            self.model_settings = ModelSettings(**self.model_settings)

            if self.model_settings.line_capacity_expansion == True and self.model_settings.line_capacity == False:
                logger.error("line_capacity_expansion setting is True but line_capacity setting is False. " \
                             "It does not make sense to expand line capacities if line capacities are neglected. " \
                             "If line_capacity_expansion is True, line_capacity must also be True." \
                             "If line_capacity_expansion is False, line_capacity can be True or False.")
                raise ValueError("Inconsistent line capacity settings")

            if self.model_settings.generator_type_costs not in ["linear", "quadratic"]:
                logger.error("generator_type_costs setting must be either 'linear' or 'quadratic'.")
                raise ValueError("Invalid value for generator_type_costs")
            
            if self.model_settings.bus_max_flow == True and self.model_settings.line_capacity_expansion == True:
                logger.error("bus_max_flow setting is True but line_capacity_expansion setting is also True. " \
                             "It does not make sense to upgrade lines but keeping the bus max flow fixed." \
                             "You may consider bus_max_flow False in this case, and just allow line capacity expansion.")
                raise ValueError("Inconsistent bus max flow settings")
            
            if self.model_settings.bus_max_flow_expansion == True and self.model_settings.line_capacity_expansion == True:
                logger.error("bus_max_flow_expansion setting is True but line_capacity_expansion setting is also True. " \
                             "It is unexpected how the model performs. Select either line_capacity_expansion or bus_max_flow_expansion to be True, not both.")
            
            if (self.model_settings.kron_equivalent_flow_constraints == True) and (self.model_settings.line_capacity == True):
                logger.error("kron_equivalent_flow_constraints setting is True but line_capacity setting is also True. " \
                             "Both setting can be false however we don't know if it makes sense to consider both the "\
                            "thermal line limits of the original system and the line limits assigned in the Kron system.")
                raise ValueError("Inconsistent thermal flow constraints in model settings")
                        
            
        logger.info(f"Model settings: {self.model_settings}")

        if self.solver_settings is None:
            self.solver_settings = SolverSettings()
        else:
            self.solver_settings = SolverSettings(**self.solver_settings)
        
        logger.info(f"Solver settings: {self.solver_settings} \n")

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
        core.construct_capacity_expansion_model(self.system, self.model, self.model_settings, self.kron_variables)

        if self.model_settings.policies is not None:
            for policy in self.model_settings.policies:
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
        
        self.model.rescaling_factor_obj = pyo.Param(initialize=1e-2)  # To reduce range of objective function values.

        self.model.obj = pyo.Objective(expr= self.model.rescaling_factor_obj * self.model.eTotalCost, sense=pyo.minimize)
        logger.info(f"> Completed in {time.time() - start_time:.2f} seconds. \n")


    def solve(self):
        """
        Solve the capacity expansion optimization model.
        """
        # Use root logger so solver output also goes to the file handler attached there
        start_time = time.time()
        logger.info("> Solving capacity expansion model...")
        solver = pyo.SolverFactory(self.solver_settings.solver_name)
        
        # Write solver output to sting_log.txt
        with capture_output(output=LogStream(logger=logging.getLogger(), level=logging.INFO)):
            results = solver.solve(self.model, options=self.solver_settings.solver_options, tee=self.solver_settings.tee)

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
        costs = pl.DataFrame({'component' : ['cost_per_timepoint_USD', 'cost_per_period_USD', 'total_cost_USD'],
                              'cost' : [  sum( pyo.value(self.model.eCostPerTp[t]) * t.weight for t in self.system.tp), 
                                            pyo.value(self.model.eCostPerPeriod), 
                                            pyo.value(self.model.eTotalCost)]})
        costs.write_csv(os.path.join(self.output_directory, 'costs_summary.csv'))

        generator.export_results_capacity_expansion(self.system, self.model, self.output_directory)
        storage.export_results_capacity_expansion(self.system, self.model, self.output_directory)
        core.export_results_capacity_expansion(self.system, self.model, self.output_directory)

        if self.model_settings.policies is not None:
            for policy in self.model_settings.policies:
                class_module = importlib.import_module(policy) 
                if hasattr(class_module, "export_results_capacity_expansion"):
                    getattr(class_module, "export_results_capacity_expansion")(self.system, self.model, self.output_directory)

    @timeit
    def inspect_coefficients(self):
        """
        Inspection of coefficients in the model.
        """

        min_coef = math.inf
        max_coef = 0

        min_coef_info = None
        max_coef_info = None

        min_rhs = math.inf
        max_rhs = -math.inf

        min_rhs_info = None
        max_rhs_info = None

        for c in self.model.component_data_objects(Constraint, active=True):
            repn = generate_standard_repn(c.body, compute_values=False)

            for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                val = abs(coef)

                if val == 0:
                    continue

                if val < min_coef:
                    min_coef = val
                    min_coef_info = (coef, c.name, var.name)

                if val > max_coef:
                    max_coef = val
                    max_coef_info = (coef, c.name, var.name)

            if c.lower is not None:
                v = abs(value(c.lower))
                if v < min_rhs:
                    min_rhs = v
                    min_rhs_info = (value(c.lower), c.name, "lower")
                if v > max_rhs:
                    max_rhs = v
                    max_rhs_info = (value(c.lower), c.name, "lower")


            if c.upper is not None:
                v = abs(value(c.upper))
                if v < min_rhs:
                    min_rhs = v
                    min_rhs_info = (value(c.upper), c.name, "upper")
                if v > max_rhs:
                    max_rhs = v
                    max_rhs_info = (value(c.upper), c.name, "upper")

        obj = next(self.model.component_data_objects(Objective, active=True))

        repn = generate_standard_repn(obj.expr, compute_values=False)

        min_obj_coef = math.inf
        max_obj_coef = 0
        min_obj_info = None
        max_obj_info = None

        for var, coef in zip(repn.linear_vars, repn.linear_coefs):

            val = abs(coef)

            if val < min_obj_coef:
                min_obj_coef = coef
                min_obj_info = (coef, var.name)

            if val > max_obj_coef:
                max_obj_coef = coef               
                max_obj_info = (coef, var.name)

        logger.info(f"  - Matrix coefficient extremes:")
        logger.info(f"     - Minimum coefficient: {min_coef_info[0]}")
        logger.info(f"       Constraint={min_coef_info[1]}  ")
        logger.info(f"       Variable={min_coef_info[2]}")
        logger.info(f"     - Maximum coefficient: {max_coef_info[0]}")
        logger.info(f"       Constraint={max_coef_info[1]}  ")
        logger.info(f"       Variable={max_coef_info[2]} \n")
        logger.info(f"  - RHS extremes:")
        logger.info(f"     - Minimum RHS: {min_rhs_info[0]}")
        logger.info(f"       Constraint={min_rhs_info[1]}  ")
        logger.info(f"       Bound={min_rhs_info[2]}")
        logger.info(f"     - Maximum RHS: {max_rhs_info[0]}")
        logger.info(f"       Constraint={max_rhs_info[1]}  ")
        logger.info(f"       Bound={max_rhs_info[2]} \n")

        logger.info(f"  - Objective coefficient extremes:")
        logger.info(f"     - Minimum coefficient: {min_obj_info[0]}")
        logger.info(f"       Variable={min_obj_info[1]}")
        logger.info(f"     - Maximum coefficient: {max_obj_info[0]}")
        logger.info(f"       Variable={max_obj_info[1]}")