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
            }

gurobi_solver_settings = {
                "solver_name": "gurobi",
                "tee": True,
                "solver_options": {'BarHomogeneous':1,
                                   'FeasibilityTol':1e-5,
                                   'CrossOver':0,
			           'Method': 2},
}

sys = main.run_capex(
    case_directory=case_dir, 
    solver_settings=mosek_solver_settings, 
    model_settings=model_settings)
#reduced_system = main.run_kron(case_dir)

capex, kr = main.run_kron_capex(
    case_directory=case_dir, 
    model_settings=model_settings, 
    solver_settings=mosek_solver_settings)
# Perform capacity expansion with Kron reduced system
#output_directory = os.path.join(case_dir, "outputs", "kron_reduction")
#capex = CapacityExpansion(
#    system=reduced_system, 
#    output_directory=output_directory, 
#    solver_settings=mosek_solver_settings, 
#    model_settings=model_settings)
#capex.solve()

# Resolve capacity expansion with the base system
#sys = main.run_capex(case_dir, solver_settings=mosek_solver_settings, model_settings=model_settings)

print('ok')
