# Import Python standard and third-party packages
from pathlib import Path
import os
# Import sting package
from sting import main
from sting.modules.capacity_expansion import CapacityExpansion

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

mosek_solver_settings = {
                "solver_name": "mosek_direct",
                "tee": True,
                "solver_options": {'MSK_DPAR_INTPNT_TOL_PFEAS':'1e-5'},
            }
model_settings = {
                "gen_costs": "quadratic",
                "consider_shedding": True,
                "consider_line_capacity": True,
                "consider_bus_max_flow": True,
            }

kron_settings = {
                "bus_neighbor_limit": None,
                "tolerance": 0.0,
                "find_kron_removable_buses": True,
                "consider_kron_removable_bus_attribute": False,
                "expand_line_capacity": False,
                "cost_fixed_power_USDperkW": 1,
                "line_capacity_method": "all"
}

kr = main.run_kron(
    case_directory=case_dir, 
    solver_settings=mosek_solver_settings, 
    kron_settings=kron_settings)


print('ok')
