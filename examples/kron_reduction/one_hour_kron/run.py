# Import Python standard and third-party packages
from pathlib import Path
import os
# Import sting package
from sting import main
from sting.modules.capacity_expansion import CapacityExpansion

# Specify path of the case study directory

if __name__ == "__main__":

    case_dir = Path(__file__).resolve().parent

    mosek_solver_settings = {
                "solver_name": "mosek_direct",
                "tee": True,
                "solver_options": {'MSK_DPAR_INTPNT_TOL_PFEAS':'1e-5'},
            }
    model_settings = {
                "generator_type_costs": "linear",
                "load_shedding": False,
                "line_capacity_expansion": False,
                "line_capacity": True,
                "kron_equivalent_flow_constraints": False,
            }

    kron_settings = {
                "bus_neighbor_limit": None,
                "tolerance": 0,
                "find_kron_removable_buses": False,
                "consider_kron_removable_bus_attribute": True,
                "expand_line_capacity": False,
                "cost_fixed_power_USDperkW": 1,
                "line_capacity_method": "NONE"
                    }

    main.run_capex(
        case_directory=case_dir, 
        model_settings=model_settings,
        solver_settings=mosek_solver_settings)

    kr = main.run_kron(
        case_directory=case_dir, 
        solver_settings=mosek_solver_settings, 
        kron_settings=kron_settings)
    
    model_settings["kron_equivalent_flow_constraints"] = True
    model_settings["line_capacity"] = False

    sys, kr = main.run_kron_capex(
        case_directory=case_dir, 
        model_settings=model_settings,
        solver_settings=mosek_solver_settings, 
        kron_settings=kron_settings)
    print('ok')
