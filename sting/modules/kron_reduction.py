# ----------------------
# Import python packages
# ----------------------
import polars as pl
import numpy as np
import os
import logging
import copy

from dataclasses import dataclass
from typing import NamedTuple
from itertools import combinations
from scipy.linalg import solve

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.line.pi_model import LinePiModel
from sting.utils.graph_matrices import build_admittance_matrix_from_lines, build_network_graph_from_lines
from sting.utils.data_tools import mat2cell, timeit, matrix_to_csv

# Logger
logger = logging.getLogger(__name__)

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

# -----------
# Main class
# -----------
@dataclass
class KronReduction():
    '''
    Kron reduction of power system networks.

    It performs Kron reduction on the admittance matrix.
    Power flow can be described by the following equations

    (1) g - d = Y_pp * θ_p + Y_pq * θ_q
    (2)     0 = Y_qp * θ_p + Y_qq * θ_q

    where θ_q are the set of buses with zero generation and load.
    They can be eliminated by solving for θ_q in (2) and substituting
    into (1). 
    '''
    system: System
    output_directory: str = None
    removable_buses: set[str] = None
    Y: np.ndarray = None
    Y_red: np.ndarray = None
    settings: KronReductionSettings = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = KronReductionSettings()
        else:
            self.settings = KronReductionSettings(**self.settings)

        self.set_output_folder()
        self.system = copy.deepcopy(self.system)
        self.build_admittance_matrix()
        if self.removable_buses is None:
            self.get_removable_buses()

    def set_output_folder(self):
        """
        Set up the output folder for storing results.
        """
        if self.output_directory is None:
            self.output_directory = os.path.join(self.system.case_directory, "outputs", "kron_reduction")
        os.makedirs(self.output_directory, exist_ok=True)

    def build_admittance_matrix(self):
        """
        Build the admittance matrix of the system.
        """
        self.Y = build_admittance_matrix_from_lines(len(self.system.bus), self.system.line_pi)

        if self.settings.print_matrices:
            bus_names = [b.name for b in self.system.bus]
            matrix_to_csv(
                filepath=os.path.join(self.output_directory, "system_conductance_matrix.csv"), 
                                  matrix=self.Y.real, 
                                  index=bus_names, columns=bus_names
            )
            matrix_to_csv(
            filepath=os.path.join(self.output_directory, "system_susceptance_matrix.csv"), 
                                  matrix=self.Y.imag, 
                                  index=bus_names, columns=bus_names
            )
            matrix_to_csv(
            filepath=os.path.join(self.output_directory, "system_absolute_admittance_matrix.csv"), 
                                  matrix=abs(self.Y), 
                                  index=bus_names, columns=bus_names
            )

    @timeit
    def get_removable_buses(self):
        """
        Identify removable buses for Kron reduction.

        Identify the name of all buses with zero generation and zero load
        to be removed via Kron reduction.
        """
        
        removable_buses = {bus.name for bus in self.system.bus}
        logger.info(f" - Total number of buses: {len(removable_buses)}")

        if self.settings.consider_kron_removable_bus_attribute == True:
            removable_buses_by_attribute = {bus.name for bus in self.system.bus if (bus.kron_removable_bus == True)}
            logger.info(f" - Buses with Kron removable attribute: {len(removable_buses_by_attribute)}")
            if len(removable_buses_by_attribute) == 0:
                logger.info(" - No buses have the 'kron_removable_bus' attribute set to True. No buses will be removed via this attribute.")
            else:
                removable_buses = removable_buses.intersection(removable_buses_by_attribute)

        if self.settings.find_kron_removable_buses == True:
            generation_buses = {gen.bus for gen in self.system.gen}
            storage_buses = {sto.bus for sto in self.system.ess}
            load_buses = {load.bus for load in self.system.load if load.load_MW > 0}
            non_removable_buses = generation_buses.union(storage_buses).union(load_buses)
            logger.info(f" - Buses with either generation, load or storage: {len(non_removable_buses)}")
            removable_buses = removable_buses - non_removable_buses

        if self.settings.bus_neighbor_limit is not None:
            Yabs = np.abs(self.Y)
            non_zero_counts = ((Yabs >= self.settings.tolerance)).sum(axis=1)
            ids = np.where( non_zero_counts <= (self.settings.bus_neighbor_limit + 1) )[0]
            buses_with_neighbors = {n.name for n in self.system.bus if n.id in ids}
            logger.info(f" - Buses with at most {self.settings.bus_neighbor_limit} neighbors: {len(buses_with_neighbors)}")
            removable_buses = removable_buses.intersection(buses_with_neighbors)
        
        self.removable_buses = removable_buses

        # Export removable buses to CSV
        df = pl.DataFrame({"bus": list(removable_buses)})
        df.write_csv(os.path.join(self.output_directory, "kron_removable_buses.csv"))

        logger.info(f" - Kron reduction will remove {len(removable_buses)} buses out of {len(self.system.bus)} total buses.")
        
    @timeit 
    def reduce(self):
        """
        Reduction of the system via Kron reduction.
        """
        # Partition all bus objects into those that will   
        # be kept and those that will be removed.
        logger.info(f" - Original system has {len(self.system.bus)} buses and {len(self.system.line_pi)} lines.")

        keep, remove = [], []
        for b in self.system.bus:
            if b.name in self.removable_buses:
                remove.append(b)
            else:
                keep.append(b)
        
        # Reorder the buses in the system so that those to
        # be removed will occur second
        self.system.bus = []
        for b in (keep + remove):
            self.system.add(b)
        
        # Update all line and generator indices
        self.system.apply("post_system_init", self.system)
        # [!] WARNING [!] Admittance matrix must be recomputed *after* permuting the buses
        self.build_admittance_matrix()
        
        # Number of total, unused (q), and real buses (p)
        n_bus = len(self.system.bus)
        q = len(self.removable_buses)
        p = n_bus - q

        # Build & partition admittance matrix
        (Y_pp, Y_pq), (Y_qp, Y_qq) = mat2cell(self.Y, [p,q], [p,q])
        # Back substitute to get reduced matrix
        invY_qq = solve(Y_qq, np.eye(q))
        Y_red = Y_pp - Y_pq @ (invY_qq) @ Y_qp
        # Setting zero connectivity between buses below the tolerance
        Y_red[abs(Y_red) < self.settings.tolerance] = 0

        # Estimate max line capacity for the new system
        G = self.get_reduced_line_capacities()

        # Remove the reduced buses from the system
        self.system.bus = self.system.bus[:p]
        # Build the new reduced lines
        self.system.line_pi = []

        # Examine non-zero/non-diagonal entries in the upper triangle of Y
        for i, j in zip(*np.triu_indices(p)):
            
            if (i == j) or (Y_red[i, j] == 0):
                continue
            
            y = -Y_red[i, j]
            z = 1/y

            bus_f = self.system.bus[i].name
            bus_t = self.system.bus[j].name
            cap_existing_power_MW = G[bus_f][bus_t]["cap_existing_power_MW"]

            line = LinePiModel(
                name=f"Y_kron_f{i}-t{j}",
                # Line connectivity
                from_bus=bus_f, from_bus_id=i,
                to_bus=bus_t, to_bus_id=j,
                # Branch & shunt parameters
                r_pu=z.real, x_pu=z.imag, # Z = R + jX
                g_pu=y.real, b_pu=y.imag, # Y = G + jB
                # Line capacity
                cap_existing_power_MW=cap_existing_power_MW,
                cost_fixed_power_USDperkW=self.settings.cost_fixed_power_USDperkW,
                expand_capacity=self.settings.expand_line_capacity
            )
            self.system.add(line)
        
        self.Y_red = Y_red

        if self.settings.print_matrices:
            self.print_reduced_admittance_matrix()

        logger.info(f" - Reduced system has {p} buses and {len(self.system.line_pi)} lines.")

    def print_reduced_admittance_matrix(self):
        """
        Print the reduced admittance matrix to CSV files.
        """
        if self.Y_red is None:
            raise ValueError("Reduced admittance matrix not available. Please run the 'reduce' method first.")
        
        bus_names = [b.name for b in self.system.bus]
        matrix_to_csv(
            filepath=os.path.join(self.output_directory, "reduced_conductance_matrix.csv"), 
                              matrix=self.Y_red.real, 
                              index=bus_names, columns=bus_names
        )
        matrix_to_csv(
        filepath=os.path.join(self.output_directory, "reduced_susceptance_matrix.csv"), 
                              matrix=self.Y_red.imag, 
                              index=bus_names, columns=bus_names
        )
        matrix_to_csv(
        filepath=os.path.join(self.output_directory, "reduced_absolute_admittance_matrix.csv"), 
                              matrix=abs(self.Y_red), 
                              index=bus_names, columns=bus_names
        )

    def get_reduced_line_capacities(self):
        """
        Estimate the maximum power flow that can be transferred between 
        two buses in the Kron reduced system. 

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
        G = build_network_graph_from_lines(self.system.bus, self.system.line_pi)
        w = "cap_existing_power_MW" # Name of edge weight

        for v in self.removable_buses:
            if v not in G:
                raise KeyError("Cannot remove a bus that does not exist in the system.")
            
            # Step 1. Get neighbors their incident edge weights
            neighbors = list(G.neighbors(v))
            edge_weights = {u: G[v][u][w] for u in neighbors}

            # Step 2. Add new lines to the system
            for i, j in combinations(neighbors, 2):
                new_weight = min(edge_weights[i], edge_weights[j])

                if G.has_edge(i, j):
                    G[i][j][w] = max(new_weight, G[i][j][w])
                    
                else:
                    G.add_edge(i, j, cap_existing_power_MW=new_weight)
            
            # Step 3. Remove the current node from the graph
            G.remove_node(v)
        return G