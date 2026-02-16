"""
This script runs the capacity expansion model for the 5-bus case study with 1-hour time resolution,
and then runs a cost production model with the built capacities from the capacity expansion model. 
The data parameters and model settings are configured so that the capacity expansion is actually cost production model.
The cost per timepoint for both runs be 17479.8969 USD.

Author: Paul Serna-Torre
Date: 2026-02-15

%   Based on data from ...
%     F.Li and R.Bo, "Small Test Systems for Power System Economic Studies",
%     Proceedings of the 2010 IEEE Power & Energy Society General Meeting

%   Created by Rui Bo in 2006, modified in 2010, 2014.
"""

# Import Python standard and third-party packages
from pathlib import Path
import os

# Import sting package
from sting import main

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Solver and model settings for the capacity expansion run
mosek_solver_settings = {
                "solver_name": "mosek_direct",
                "tee": True,
                "solver_options": {},
            }
model_settings = {
        "generator_type_costs": "quadratic",
        "load_shedding": False,
        "generation_capacity_expansion": True,
        "storage_capacity_expansion": False,
        "line_capacity_expansion": False,
        "line_capacity": True,
        "kron_equivalent_flow_constraints": False,
        "bus_max_flow_expansion": False,
        "bus_max_flow": False
    }

gurobi_solver_settings = {
                "solver_name": "gurobi",
                "tee": True,
                "solver_options": {'BarHomogeneous':1,
                                   'FeasibilityTol':1e-5,
                                   'CrossOver':0,
			           'Method': 2},
}

# Run capacity expansion
main.run_capex(case_dir, solver_settings=mosek_solver_settings, model_settings=model_settings)


# Solver and model settings for the cost production run with built capacities from the capacity expansion run
model_settings = {
        "generator_type_costs": "linear",
        "load_shedding": False,
        "generation_capacity_expansion": False, # No expansion in generation
        "storage_capacity_expansion": False, # No expansion in storage
        "line_capacity_expansion": False, # No expansion in lines
        "line_capacity": True,
        "kron_equivalent_flow_constraints": False,
        "bus_max_flow_expansion": False, # No expansion in bus max flow
        "bus_max_flow": False,
    }

# Run cost production with built capacities from the capacity expansion run
capex, _ = main.run_capex_with_initial_build(case_dir, solver_settings=gurobi_solver_settings, model_settings=model_settings,
                                             output_directory=os.path.join(case_dir, "outputs", "cost_production"),
                                             built_capacity_directory=os.path.join(case_dir, "outputs", "capacity_expansion"), make_non_expandable=True)

# This step is just to check the inputs that the previous run considered,
# It is not necessary
capex.system.write_csv(output_directory= os.path.join(case_dir, "outputs", "input_data_for_cost_production"), types = [int, float, str, bool])

print('ok')