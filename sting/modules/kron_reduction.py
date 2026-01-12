# ----------------------
# Import python packages
# ----------------------
from dataclasses import dataclass
import copy
import numpy as np
from scipy.linalg import solve
import time
import logging

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.line.pi_model import LinePiModel
from sting.utils.graph_matrices import build_admittance_matrix_from_lines
from sting.utils.data_tools import mat2cell, timeit

logger = logging.getLogger(__name__)

# -----------
# Main class
# -----------
@dataclass
class KronReduction():
    '''
    Power flow can be described by the following equations

    (1) g - d = Y_pp * θ_p + Y_pq * θ_q
    (2)     0 = Y_qp * θ_p + Y_qq * θ_q

    where θ_q are the set of buses with zero generation and load.
    They can be eliminated by solving for θ_q in (2) and substituting
    into (1). 
    '''
    system: System
    remove_buses: set = None
    
    def __post_init__(self):
        self.system = copy.deepcopy(self.system)
        if self.remove_buses is None:
            self.get_remove_buses()

    def get_remove_buses(self):
        """
        Identify the name of all buses with zero generation and zero load
        to be removed via Kron reduction.
        """

        all_buses = set([bus.name for bus in self.system.bus])
        generation_buses = set([gen.bus for gen in self.system.gen])
        storage_buses = set([sto.bus for sto in self.system.ess])
        load_buses = set([load.bus for load in self.system.load if load.load_MW > 0])
        self.remove_buses = all_buses - generation_buses.union(storage_buses).union(load_buses)

        Y = build_admittance_matrix_from_lines(len(self.system.bus), self.system.line_pi)
        B = Y.imag
        N = self.system.bus
        N_at_bus = {n.id: [N[k] for k in np.nonzero(B[n.id, :])[0] if k != n.id] for n in N}
        N_at_bus ={k: v for k, v in N_at_bus.items() if len(v) <= 1000}

        self.remove_buses = self.remove_buses.intersection(set([N[k].name for k in N_at_bus.keys()]))


        logger.info(f"> Kron reduction will remove {len(self.remove_buses)} buses out of {len(all_buses)} total buses. \n")
        
    @timeit 
    def reduce(self):
        # Partition all bus objects into those that will   
        # be kept and those that will be removed.
        keep, remove = [], []
        for b in self.system.bus:
            if b.name in self.remove_buses:
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
        q = len(self.remove_buses)
        p = n_bus - q

        # Build & partition admittance matrix
        Y = build_admittance_matrix_from_lines(n_bus, self.system.line_pi)
        (Y_pp, Y_pq), (Y_qp, Y_qq) = mat2cell(Y, [p,q], [p,q])
        # Back substitute to get reduced matrix
        invY_qq = solve(Y_qq, np.eye(q))
        Y_red = Y_pp - Y_pq @ (invY_qq) @ Y_qp

        # Remove the reduced buses from the system
        self.system.bus = self.system.bus[:p]
        # Build the new reduced lines
        self.system.line_pi = []

        for i, j in zip(*np.triu_indices(p)):
            # Skip all unconnected nodes and the diagonal
            if (i == j) or (np.isclose(abs(Y_red[i, j]), 0,  atol=1e-4)):
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
        logging.info("> Kron reduction completed. \n")

    def line_cap():
        # Create graph object
        #    - TODO: https://ieeexplore.ieee.org/abstract/document/6506059

        # for each bus to remove:
        #.  1. look up buses nearest neighbors
        #.  3. Create new edges between *all* neighbors with a weight given by min of both edges
        #.  4. Delete the bus.
        pass