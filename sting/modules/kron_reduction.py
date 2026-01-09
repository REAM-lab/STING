# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.utils.graph_matrices import build_oriented_incidence_matrix, build_admittance_matrix_from_lines
from sting.utils.data_tools import mat2cell
import numpy as np
import sting.system.selections as sl
import copy

class KronReduction():
    system: System
    remove_buses: set = None
    
    def __postinit__(self):
        self.system = copy.deepcopy(self.system)

    def reduce(self):

        def sort_function(item):
            if item in self.remove_buses:
                return (1, item)
            else:
                return (0, item)
        
        # Sort all system buses placing buses to remove last
        self.system.bus.sort(key=sort_function)
        # Update all indices
        self.system.apply("assign_indices", self.system)

        n_bus = len(self.system.bus)
        q = len(self.remove_buses)
        p = n_bus - q

        from_bus, to_bus = self.system.query(sl.lines()).select("from_bus_id", "to_bus_id")
        branch_data = list(zip(from_bus, to_bus))

        # g - d = (G' * Y * G) z
        #   g: generation
        #   d: demand
        #   z: bus angle (relative to slack)
        G = build_oriented_incidence_matrix(n_bus, branch_data) # dims: (n_bus, n_branch)
        Y = build_admittance_matrix_from_lines(n_bus, self.system.lines) # dims: (n_bus, n_bus)
        Y = np.diag(np.diag(Y))

        # Partition G and Y
        G = mat2cell(G, [p, q], [p, q])
        Y = mat2cell(G, [p, q], [p, q])

        
        P, Q = Y[1,1], Y[2,2]
        A, B, C, D = G[1,1], G[1,2], G[2,1], G[2,2]
        
        alpha = A.T @ P @ A + C.T @ Q @ C
        beta = A.T @ P @ B + C.T @ Q @ D
        delta = 
        
        return

    def to_system():
        pass