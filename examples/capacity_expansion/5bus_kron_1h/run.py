# Import Python standard and third-party packages
from pathlib import Path
import os
# Import sting package
from sting import main
from sting.modules.capacity_expansion import CapacityExpansion

if __name__ == "__main__":
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

    gurobi_solver_settings = {
                    "solver_name": "gurobi",
                    "tee": True,
                    "solver_options": {'BarHomogeneous':1,
                                    'FeasibilityTol':1e-5,
                                    'CrossOver':0,
                        'Method': 2},
    }

    kron_settings = {
                    "bus_neighbor_limit": None,
                    "tolerance": 0.0,
                    "find_kron_removable_buses": True,
                    "consider_kron_removable_bus_attribute": False,
                    "expand_line_capacity": False,
                    "cost_fixed_power_USDperkW": 1,
                    "line_capacity_method": "optimization",
                    "max_workers": 20
    }

    #sys = main.run_capex(
    #    case_directory=case_dir, 
    #    solver_settings=mosek_solver_settings, 
    #    model_settings=model_settings)

    capex, kr = main.run_kron_capex(
        case_directory=case_dir, 
        model_settings=model_settings, 
        solver_settings=mosek_solver_settings,
        kron_settings=kron_settings
        )


    print('ok')
