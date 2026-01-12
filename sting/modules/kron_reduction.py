# ----------------------
# Import python packages
# ----------------------
from dataclasses import dataclass
import copy
import numpy as np
from scipy.linalg import solve
import os
import logging
from typing import NamedTuple

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.line.pi_model import LinePiModel
from sting.utils.graph_matrices import build_admittance_matrix_from_lines
from sting.utils.data_tools import mat2cell, timeit, matrix_to_csv

# Logger
logger = logging.getLogger(__name__)

# ----------
# Sub-classes
# ----------
class KronReductionSettings(NamedTuple):
    print_matrices: bool = True
    tolerance: float = 1e-4
    bus_neighbor_limit: int = None

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

    @timeit
    def get_removable_buses(self):
        """
        Identify removable buses for Kron reduction.

        Identify the name of all buses with zero generation and zero load
        to be removed via Kron reduction.
        """

        all_buses = set([bus.name for bus in self.system.bus])
        logger.info(f" - Total number of buses: {len(all_buses)}")

        generation_buses = set([gen.bus for gen in self.system.gen])
        storage_buses = set([sto.bus for sto in self.system.ess])
        load_buses = set([load.bus for load in self.system.load if load.load_MW > 0])
        non_removable_buses = generation_buses.union(storage_buses).union(load_buses)
        logger.info(f" - Buses with either generation, load or storage: {len(non_removable_buses)}")

        removable_buses = all_buses - non_removable_buses

        if self.settings.bus_neighbor_limit is not None:
            Yabs = np.abs(self.Y)
            non_zero_counts = ((~np.isclose(Yabs, 0, atol=self.settings.tolerance))).sum(axis=1)
            ids = np.where( non_zero_counts <= (self.settings.bus_neighbor_limit + 1) )[0]
            buses_with_neighbors = set([n.name for n in self.system.bus if n.id in ids])

            logger.info(f" - Buses with at most {self.settings.bus_neighbor_limit} neighbors: {len(buses_with_neighbors)}")

            removable_buses = removable_buses.intersection(buses_with_neighbors)
        
        self.removable_buses = removable_buses
        logger.info(f" - Kron reduction will remove {len(removable_buses)} buses out of {len(all_buses)} total buses.")
        
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
        
        # Number of total, unused, and real buses
        n_bus = len(self.system.bus)
        q = len(self.removable_buses)
        p = n_bus - q

        # Build & partition admittance matrix
        (Y_pp, Y_pq), (Y_qp, Y_qq) = mat2cell(self.Y, [p,q], [p,q])
        # Back substitute to get reduced matrix
        invY_qq = solve(Y_qq, np.eye(q))
        
        Y_red = Y_pp - Y_pq @ (invY_qq) @ Y_qp

        # Remove the reduced buses from the system
        self.system.bus = self.system.bus[:p]
        # Build the new reduced lines
        self.system.line_pi = []

        for i, j in zip(*np.triu_indices(p)):
            # Skip all unconnected nodes and the diagonal
            if (i == j) or (np.isclose(abs(Y_red[i, j]), 0,  atol=self.settings.tolerance)):
                continue
            
            y = -Y_red[i, j]
            z = 1/y

            line = LinePiModel(
                name=f"Y_kron_{i}{j}",
                # Line connectivity
                from_bus=self.system.bus[i].name, from_bus_id=i,
                to_bus=self.system.bus[j].name, to_bus_id=j,
                # Line parameters
                r_pu=z.real, x_pu=z.imag, # Z = R + jX
                g_pu=y.real, b_pu=y.imag, # Y = G + jB
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

    def line_cap():
        # Create graph object
        #    - TODO: https://ieeexplore.ieee.org/abstract/document/6506059

        # for each bus to remove:
        #.  1. look up buses nearest neighbors
        #.  3. Create new edges between *all* neighbors with a weight given by min of both edges
        #.  4. Delete the bus.
        pass