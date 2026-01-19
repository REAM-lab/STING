# ----------------------
# Import python packages
# ----------------------
import time
import polars as pl
import numpy as np
import os
import logging
import copy

from concurrent.futures import ProcessPoolExecutor

from dataclasses import dataclass, field
from typing import NamedTuple
from itertools import combinations
from scipy.linalg import solve
import pyomo.environ as pyo
from pyomo.common.log import LogStream
from pyomo.common.tee import capture_output

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.line.pi_model import LinePiModel
from sting.utils.graph_matrices import build_admittance_matrix_from_lines, build_network_graph_from_lines
from sting.utils.data_tools import mat2cell, timeit, matrix_to_csv

# Logger
logger = logging.getLogger(__name__)
# Global read only KronReduction object shared
# by workers while multiprocessing optimizations.
SELF = None

# ----------
# Sub-classes
# ----------
class KronReductionSettings(NamedTuple):
    """
    Settings for Kron reduction.
    """
    print_matrices: bool = True
    tolerance: float = 1e-4
    bus_neighbor_limit: int = None
    find_kron_removable_buses: bool = True
    consider_kron_removable_bus_attribute: bool = False
    # Should lines created by Kron reduction be expandable?
    # If so what is the fixed cost of upgrading a line?
    expand_line_capacity: bool = False
    cost_fixed_power_USDperkW: float = None
    line_capacity_method: str = None
    max_workers: int = None

class SolverSettings(NamedTuple):
    """
    Solver settings for optimization problems.
    """
    solver_name: str = "mosek_direct"
    tee: bool = True
    solver_options: dict = field(default_factory=dict)

# -----------
# Main class
# -----------
@dataclass
class KronReduction():
    '''
    Kron reduction of DC power system networks.

    It performs Kron reduction on the admittance matrix.
    Power flow can be described by the following equations

    (1) g - d = B_pp * θ_p + B_pq * θ_q
    (2)     0 = B_qp * θ_p + B_qq * θ_q

    where θ_q are the set of buses with zero generation and load.
    They can be eliminated by solving for θ_q in (2) and substituting
    into (1). 
    '''
    original_system: System
    kron_system: System = None
    output_directory: str = None
    removable_buses: set[str] = None
    Y_original: np.ndarray = None
    Y_kron: np.ndarray = None
    settings: KronReductionSettings = None
    solver_settings: SolverSettings = None
    
    def __post_init__(self):
        logger.info("\n>> Starting Kron reduction ...\n")

        logger.info(f"{self.settings}")
        if self.settings is None:
            self.settings = KronReductionSettings()
        else:
            self.settings = KronReductionSettings(**self.settings)

        self.set_output_folder()
        self.original_system = copy.deepcopy(self.original_system)
        if self.removable_buses is None:
            self.get_removable_buses()

    def set_output_folder(self):
        """
        Set up the output folder for storing results.
        """
        if self.output_directory is None:
            self.output_directory = os.path.join(self.original_system.case_directory, "outputs", "kron_reduction")
        os.makedirs(self.output_directory, exist_ok=True)

    def build_admittance_matrix(self):
        """
        Build the admittance matrix of the system.
        """
        self.Y_original = build_admittance_matrix_from_lines(len(self.original_system.bus), self.original_system.line_pi)

        if self.settings.print_matrices:
            bus_names = [b.name for b in self.original_system.bus]
            matrix_to_csv(
                filepath=os.path.join(self.output_directory, "original_system_conductance_matrix.csv"), 
                                  matrix=self.Y_original.real, 
                                  index=bus_names, columns=bus_names
            )
            matrix_to_csv(
            filepath=os.path.join(self.output_directory, "original_system_susceptance_matrix.csv"), 
                                  matrix=self.Y_original.imag, 
                                  index=bus_names, columns=bus_names
            )
            matrix_to_csv(
            filepath=os.path.join(self.output_directory, "original_system_absolute_admittance_matrix.csv"), 
                                  matrix=abs(self.Y_original), 
                                  index=bus_names, columns=bus_names
            )

    @timeit
    def get_removable_buses(self):
        """
        Identify removable buses for Kron reduction.

        Identify the name of all buses with zero generation and zero load
        to be removed via Kron reduction.
        """
        
        removable_buses = {bus.name for bus in self.original_system.bus}
        logger.info(f" - Total number of buses: {len(removable_buses)}")

        if self.settings.consider_kron_removable_bus_attribute == True:
            removable_buses_by_attribute = {bus.name for bus in self.original_system.bus if (bus.kron_removable_bus == True)}
            logger.info(f" - Buses with Kron removable attribute: {len(removable_buses_by_attribute)}")
            if len(removable_buses_by_attribute) == 0:
                logger.info(" - No buses have the 'kron_removable_bus' attribute set to True. No buses will be removed via this attribute.")
            else:
                removable_buses = removable_buses.intersection(removable_buses_by_attribute)

        if self.settings.find_kron_removable_buses == True:
            generation_buses = {gen.bus for gen in self.original_system.gen}
            storage_buses = {sto.bus for sto in self.original_system.ess}
            load_buses = {load.bus for load in self.original_system.load if np.abs(load.load_MW) != 0}
            non_removable_buses = generation_buses.union(storage_buses).union(load_buses)
            logger.info(f" - Buses with either generation, load or storage: {len(non_removable_buses)}")
            removable_buses = removable_buses - non_removable_buses

        if self.settings.bus_neighbor_limit is not None:
            self.build_admittance_matrix()
            Yabs = np.abs(self.Y_original)
            non_zero_counts = ((Yabs >= self.settings.tolerance)).sum(axis=1)
            ids = np.where( non_zero_counts <= (self.settings.bus_neighbor_limit + 1) )[0]
            buses_with_neighbors = {n.name for n in self.original_system.bus if n.id in ids}
            logger.info(f" - Buses with at most {self.settings.bus_neighbor_limit} neighbors: {len(buses_with_neighbors)}")
            removable_buses = removable_buses.intersection(buses_with_neighbors)
        
        self.removable_buses = removable_buses

        # Export removable buses to CSV
        df = pl.DataFrame({"bus": list(removable_buses)})
        df.write_csv(os.path.join(self.output_directory, "kron_removable_buses.csv"))

        logger.info(f" - Kron reduction will remove {len(removable_buses)} buses out of {len(self.original_system.bus)} total buses.")

    def reorder_indices_in_original_system(self):
        """
        Reorder the indices of buses, lines, generators, and loads
        in the original system so that the buses to be removed are last.
        """
        keep, remove = [], []
        for b in self.original_system.bus:
            if b.name in self.removable_buses:
                remove.append(b)
            else:
                keep.append(b)
        
        # Reorder the buses in the system so that those to
        # be removed will occur second
        self.original_system.bus = []
        for b in (keep + remove):
            self.original_system.add(b)
        
        # Update all line and generator indices
        self.original_system.apply("post_system_init", self.original_system)

        # [!] WARNING [!] Admittance matrix must be recomputed *after* permuting the buses
        self.build_admittance_matrix()
        
    @timeit 
    def reduce(self):
        """
        Reduction of the system via Kron reduction.
        """
        # Partition all bus objects into those that will   
        # be kept and those that will be removed.
        logger.info(f" - Original system has {len(self.original_system.bus)} buses and {len(self.original_system.line_pi)} lines.")
        
        self.reorder_indices_in_original_system()

        # Number of total, unused (q), and real buses (p)
        n_bus = len(self.original_system.bus)
        q = len(self.removable_buses)
        p = n_bus - q

        # Build & partition admittance matrix
        (Y_pp, Y_pq), (Y_qp, Y_qq) = mat2cell(self.Y_original.imag, [p,q], [p,q])
        # Back substitute to get reduced matrix
        invY_qq = solve(Y_qq, np.eye(q))
        Y_kron = Y_pp - Y_pq @ (invY_qq) @ Y_qp
        # Setting zero connectivity between buses below the tolerance
        Y_kron[abs(Y_kron) <= self.settings.tolerance] = 0

        # Rebuild the Kron system
        self.kron_system = copy.deepcopy(self.original_system)
        # Remove the reduced buses from the system
        self.kron_system.bus = self.kron_system.bus[:p]
        # Build the new reduced lines
        self.kron_system.line_pi = []

        # Examine non-zero/non-diagonal entries in the upper triangle of Y
        for i, j in zip(*np.triu_indices(p)):
            
            if (i == j) or (Y_kron[i, j] == 0):
                continue
            
            y = Y_kron[i, j]
            z = 1/y

            bus_f = self.kron_system.bus[i].name
            bus_t = self.kron_system.bus[j].name

            line = LinePiModel(
                name=f"Y_kron_f{bus_f}-t{bus_t}",
                # Line connectivity
                from_bus=bus_f, from_bus_id=i,
                to_bus=bus_t, to_bus_id=j,
                # Branch & shunt parameters
                r_pu=0, x_pu=z, # Z = R + jX
                g_pu=0, b_pu=0, # Y = G + jB
                cost_fixed_power_USDperkW=self.settings.cost_fixed_power_USDperkW,
                expand_capacity=self.settings.expand_line_capacity
            )
            self.kron_system.add(line)
        
        self.Y_kron = Y_kron

        if self.settings.print_matrices:
            self.print_reduced_admittance_matrix()

        logger.info(f" - Kron system has {p} buses and {len(self.kron_system.line_pi)} lines.")
            
        match self.settings.line_capacity_method:
            case "algorithm":
                self.assign_line_capacities_via_algorithm()
            case "optimization":
                self.assign_line_capacities_via_optimization()
            case "all":
                self.assign_line_capacities_via_algorithm()
                self.assign_line_capacities_via_optimization()
            case _:
                logger.info(" - No line capacity assignment method selected. Skipping line capacity assignment.")


    def print_reduced_admittance_matrix(self):
        """
        Print the reduced admittance matrix to CSV files.
        """
        if (self.Y_kron is None) or (self.kron_system is None):
            raise ValueError("Reduced admittance matrix not available. Please run the reduce the system first.")
        
        bus_names = [b.name for b in self.kron_system.bus]
        matrix_to_csv(
            filepath=os.path.join(self.output_directory, "reduced_susceptance_matrix.csv"), 
                              matrix=self.Y_kron, 
                              index=bus_names, columns=bus_names)

    @timeit
    def assign_line_capacities_via_algorithm(self):
        """
        Estimation of line capacities for the Kron system via algorithm.

        Nodes are eliminated iteratively through the following steps
            1.  Get the node's neighbors. 
            2a. For each unique pair of neighbors, add a new edge to the 
                graph between node_i and node_j with an edge weight of 
                min(weight_i, weight_j). 
            2b. If an edge between node_i and node_j already exists 
                combine the new edge and existing edge (which are in 
                parallel) by taking the max of their edge weights. 
            3.  After iterating over all pairs of neighbors, delete the 
                current node and process the next node to remove.

        Warning: The following algorithm is not guaranteed to be deterministic. 
            The order of node elimination *may* impact results.
                
        Note: This is a heuristic method for getting line capacities
            and is most accurate when the degree of nodes being removed is 
            less than or equal to 3. A more precise method can be found in 
            https://ieeexplore.ieee.org/abstract/document/6506059.
        """
        # A graph network of buses and lines, where edge weights are
        # the existing line capacity limits 

        G_original = build_network_graph_from_lines(self.original_system.bus, self.original_system.line_pi)
        w = "cap_existing_power_MW" # Name of edge weight

        for v in self.removable_buses:
            if v not in G_original:
                raise KeyError("Cannot remove a bus that does not exist in the system.")
            
            # Step 1. Get neighbors their incident edge weights
            neighbors = list(G_original.neighbors(v))
            edge_weights = {u: G_original[v][u][w] for u in neighbors}

            # Step 2. Add new lines to the system
            for i, j in combinations(neighbors, 2):
                new_weight = min(edge_weights[i], edge_weights[j])

                if G_original.has_edge(i, j):
                    G_original[i][j][w] = max(new_weight, G_original[i][j][w])
                    
                else:
                    G_original.add_edge(i, j, cap_existing_power_MW=new_weight)
            
            # Step 3. Remove the current node from the graph
            G_original.remove_node(v)
        
        # Update line capacities in the Kron system
        model_solution = {'from_bus': [], 'to_bus': [], 'calculated_capacity_MW': []}
        
        for line in self.kron_system.line_pi:
            i = line.from_bus
            j = line.to_bus
            line.cap_existing_power_MW = G_original[i][j][w]
            model_solution['from_bus'].append(i)
            model_solution['to_bus'].append(j)
            model_solution['calculated_capacity_MW'].append(line.cap_existing_power_MW)

        df_solution = pl.DataFrame(model_solution)
        df_solution.write_csv(os.path.join(self.output_directory, "kron_line_capacity_via_algorithm.csv"))
        logger.info(" - Exported line capacities calculated via algorithm as CSV.")
    
    @timeit
    def assign_line_capacities_via_optimization(self):
        """
        Estimation of line capacities for the Kron system via optimization.
        """

        if self.solver_settings is None:
            self.solver_settings = SolverSettings()
        else:
            self.solver_settings = SolverSettings(**self.solver_settings)

        logger.info(f"{self.solver_settings}")

        from_bus = [line.from_bus for line in self.kron_system.line_pi]
        to_bus = [line.to_bus for line in self.kron_system.line_pi]

        # Run all optimizations to determine line capacity in parallel
        with ProcessPoolExecutor(max_workers=self.settings.max_workers, initializer=worker_init, initargs=(self,)) as ex:
            calculated_capacity_MW = list(ex.map(line_capacities_optimization, from_bus, to_bus))

        model_solution = {
            'from_bus': from_bus, 
            'to_bus': to_bus, 
            'calculated_capacity_MW': calculated_capacity_MW}

        # Update line capacity in the Kron system
        for kron_line, capacity in zip(self.kron_system.line_pi, calculated_capacity_MW):
            kron_line.cap_existing_power_MW = capacity

        df_solution = pl.DataFrame(model_solution)
        df_solution.write_csv(os.path.join(self.output_directory, "kron_line_capacity_via_optimization.csv"))
        logger.info(" - Exported line capacities calculated via optimization as CSV.")

def line_capacities_optimization(i, j):

    N = SELF.original_system.bus
    L = SELF.original_system.line_pi

    #logger.info(f"> Estimating capacity for line from bus {i} to bus {j}.")
    #logger.info(" - Building optimization model ...")
    model = pyo.ConcreteModel()
    #logger.info(" - Decision variables of bus angles.")
    model.vTHETA = pyo.Var(N, within=pyo.Reals)

    slack_bus = next(n for n in N if n.bus_type == 'slack')
    model.vTHETA[slack_bus].fix(0.0)

    model.vALPHA = pyo.Var(within=pyo.NonNegativeReals)
    model.vBETA = pyo.Var(within=pyo.NonNegativeReals)
    #logger.info(f"   Size: {len(model.vTHETA) + 1} variables.")
    #logger.info(" - Constraints of angle differences for original system.")
    model.cFlowPerExpLine = pyo.Constraint(L, rule=lambda m, l: 
        (-l.cap_existing_power_MW ,100 * l.x_pu / (l.x_pu**2 + l.r_pu**2) * (m.vTHETA[N[l.from_bus_id]] - m.vTHETA[N[l.to_bus_id]]), 
        l.cap_existing_power_MW)
    )
    #logger.info(f"   Size: {len(model.cFlowPerExpLine)} constraints.")
    #logger.info(" - Constraint of angle difference for a line in the Kron system.")           
    Ni = next((bus for bus in N if bus.name == i))
    Nj = next((bus for bus in N if bus.name == j))
    from_bus_id_kron = next((n for n in SELF.kron_system.bus if n.name == i)).id
    to_bus_id_kron = next((n for n in SELF.kron_system.bus if n.name == j)).id
    y = SELF.Y_kron[from_bus_id_kron, to_bus_id_kron]
    model.cUpperDiffKronLine = pyo.Constraint(rule= lambda m:
                                100 * abs(y) * (m.vTHETA[Ni] - m.vTHETA[Nj]) == m.vALPHA - m.vBETA)

    # Neighbors at any bus
    B_original = SELF.Y_original.imag
    N_at_bus = {n.id: [N[k] for k in np.nonzero(B_original[n.id, :])[0] if k != n.id] for n in N}
    Nremovable = {n for n in SELF.original_system.bus if n.name in SELF.removable_buses}
    # FlowIn == FlowOut for all removable buses
    model.eFlowAtBus = pyo.Expression(N,expr=lambda m, n: 100 * pyo.quicksum(B_original[n.id, k.id] * (m.vTHETA[n] - m.vTHETA[k]) for k in N_at_bus[n.id]) )
    model.cNetFlowAtRemovableBus = pyo.Constraint(Nremovable, expr=lambda m, n: m.eFlowAtBus[n] == 0 )
    
    #logger.info(f"   Size: {len(model.cNetFlowAtRemovableBus)} constraints.")
    #logger.info(" - Objective function to maximize the angle difference bound.")
    model.oMaximizeAngleDiff = pyo.Objective(expr= lambda m: m.vALPHA - m.vBETA, sense=pyo.maximize)

    #start_time = time.time()
    #logger.info("> Solving optimization model to find maximum angle difference bound...")
    solver = pyo.SolverFactory(SELF.solver_settings.solver_name)

    # Write solver output to sting_log.txt
    with capture_output(output=LogStream(logger=logging.getLogger(), level=logging.INFO)):
        results = solver.solve(model, options=SELF.solver_settings.solver_options, tee=False)

    #logger.info(f"> Time spent by solver: {time.time() - start_time:.2f} seconds.")
    if results.solver.status.name != 'ok':
        logger.info(f"WARNING: Solver finished with status: {results.solver.status}, termination condition: {results.solver.termination_condition}")
    #logger.info(f"> Solver finished with status: {results.solver.status}, termination condition: {results.solver.termination_condition}.")
    #logger.info(f"> Objective value: {(pyo.value(model.oMaximizeAngleDiff)):.2f}.")
    #logger.info(f"> Time spent by solver: {time.time() - start_time:.2f} seconds.")

    return pyo.value(model.oMaximizeAngleDiff)


def worker_init(self:KronReduction):
    global SELF
    SELF = self