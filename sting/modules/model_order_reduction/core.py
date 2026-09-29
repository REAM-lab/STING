from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from scipy.linalg import block_diag
#from sting.system.core import System
from sting.utils.component_connections import get_ccm_matrices, build_ccm_permutation
from sting.utils.matrix_tools import block_permute
from sting.utils.dynamical_systems import DynamicalVariables, StateSpaceModel, QuadraticBilinearModel


type_to_class = {'ssm':StateSpaceModel, 'qbm':QuadraticBilinearModel}

@dataclass
class ModelReducer:
    """Wrapper class for creating reduced order models."""
    models: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    F: np.ndarray = None
    G: np.ndarray = None
    H: np.ndarray = None
    L: np.ndarray = None
    model_type: str = None

    @classmethod
    def from_system(cls, system, model_type) -> 'ModelReducer':
        """
        Returns a ModelReducer from system object.
        
        model_type: Either 'ssm' for StateSpaceModel or 'qbm' for QuadraticBilinearModel
        """
        # Access all component models
        components = system.query(["ccm_generators", "ccm_shunts", "ccm_branches"]).to_list()
        models = [getattr(c, model_type) for c in components]
        zones =[c.zone for c in components]

        # Construct CCM matrices
        F, G, H, L = get_ccm_matrices(system, attribute=model_type, dimI=2)
        # Permute the F and G 
        T_gen = build_ccm_permutation(system, attribute=model_type, tag="ccm_generator")
        T_sh = build_ccm_permutation(system, attribute=model_type, tag="ccm_shunt")
        T_br = build_ccm_permutation(system, attribute=model_type, tag="ccm_branch")
        T = block_diag(T_gen, T_sh, T_br)

        F = T @ F
        G = T @ G       

        return ModelReducer(models, zones, F, G, H, L, model_type=model_type)

    @property
    def inputs(self):
        return sum([m.u for m in self.models], DynamicalVariables(name=[]))

    @property 
    def outputs(self):
        return sum([m.y for m in self.models], DynamicalVariables(name=[]))

    @property
    def connections(self):
        if self.model_type == "ssm":
            return (self.F, self.G, self.H, self.L)
        if self.model_type == "qbm":
            return (self.F, self.G, self.H, self.L, None, None)

    def sort(self):
        """
        Returns a ModelReducer object where all components are  
        sorted by their associated zone.
        """
        # Sorted ids for every component
        ids, zones = zip(*sorted(zip(range(len(self.zones)), self.zones), key=lambda x: (1, x[1]) if (x[1] is not None) else (0, "")))

        # Total number of inputs/outputs for each component 
        y_stack = [len(m.y) for m in self.models]
        u_stack = [len(m.u) for m in self.models]

        # Number input/outputs for each component at the system-level.
        # We assume component and system-level outputs are the same.
        y_system = y_stack 
        u_system = [m.u.n_device for m in self.models]

        # Permute each component connection matrix to correspond to
        # the sorted components
        F = block_permute(self.F, u_stack,  y_stack,  ids)
        G = block_permute(self.G, u_stack,  u_system, ids)
        H = block_permute(self.H, y_system, y_stack,  ids)
        L = block_permute(self.L, y_system, u_system, ids)

        # And sort all the components
        models = [self.models[i] for i in ids]

        return ModelReducer(models, zones, F, G, H, L, self.model_type)


    def create_zonal_models(self):
        """
        Returns a ModelReducer object where all models within  
        a given zone have been interconnected. 
        """
        # Sort all components by their zone
        sys = self.sort()
    
        # Map each zone to the list of models in that zone
        zone_model_map = defaultdict(ModelReducer)
        for z, m in zip(sys.zones, sys.models):
            zone_model_map[z].models.append(m)

        # Number of subsystems/zones
        s = len(zone_model_map)
        # For each subsystem create a range to index u_stack and y_stack in F, G, and H
        y_stack = np.cumsum([0] + [len(m.outputs) for m in zone_model_map.values()])
        y = [range(y_stack[i-1], y_stack[i]) for i in range(1, s+1)]
        u_stack = np.cumsum([0] + [len(m.inputs) for m in zone_model_map.values()])
        u = [range(u_stack[i-1], u_stack[i]) for i in range(1, s+1)]
    
        # Select the block diagonal elements of F corresponding to each subsystem
        diagF = [sys.F[u[i], :][:, y[i]] for i in range(s)]
        Z = sys.F - block_diag(*diagF)
        # Compute the inter-subsystem connection matrix
        m, p = sys.F.shape
        # Set of indices for which there are either device level inputs/outputs 
        # or inter-subsystem level inputs/outputs
        w = np.unique(np.nonzero(np.hstack((sys.G, Z)))[0])
        v = np.unique(np.nonzero(np.vstack((sys.H, Z)))[1])
    
        # Define selection matrices Phi and Psi (from Lemma 1)
        X = np.zeros((len(w), m)) 
        X[range(len(w)), w] = 1
    
        Y = np.zeros((p, len(v)))
        Y[v, range(len(v))] = 1

        # List of all new zonal models
        zonal_models = []
    
        for i, zone_data in enumerate(zone_model_map.values()):
            u_i, y_i, F_i = u[i], y[i], diagF[i]
           
            # Matrices used to connect components within each zone
            X_i = X[[j for j, k in enumerate(w) if k in u_i], :][:, u_i]
            Y_i = Y[:, [j for j, k in enumerate(v) if k in y_i]][y_i, :]
    
            # Interconnection matrices for models within the zone
            zone_data.F = F_i
            zone_data.G = X_i.T
            zone_data.H = Y_i.T
            zone_data.L = np.zeros((Y_i.shape[1], X_i.shape[0]))
            # In
            zone_data.model_type = sys.model_type

            # Only select inputs that are device-level OR from other subsystems
            inputs = zone_data.inputs[[j for j, k in enumerate(u_i) if k in w]] 

            # Interconnect all zone models to create a single zonal model
            zone_model = zone_data.interconnect(u=inputs, y=zone_data.outputs)
            zonal_models.append(zone_model)    

        # Update the system-level interconnection matrices (to remove intrazonal connections)
        diagF, diagG, diagH, _ = zip(*[s.connections[:4] for s in zone_model_map.values()])
        W = block_diag(*diagF)
        X = block_diag(*diagG).T
        Y = block_diag(*diagH).T
        
        F = X @ (sys.F - W) @ Y
        G = X @ sys.G
        H = sys.H @ Y

        return ModelReducer(zonal_models, list(zone_model_map.keys()), F, G, H, sys.L, sys.model_type)
    

    def reduce_zonal_models(self, reducers:dict, shift_to_equilibrium=False):
        """
        Returns a ModelReducer object where each zone, with a 
        value in the supplied reducers dict, has been reduced to a specified order.
        """
        if len(self.zones) > len(set(self.zones)):
            raise TypeError("Zonal models have not been constructed, please call " \
            "`create_zonal_models` prior to reduction.")

        if shift_to_equilibrium:
            models = [m.shift_to_equilibrium() for m in self.models]
        else:
            models = self.models

        reduced_models = []
        for zone, model in zip(self.zones, models):
            if zone in reducers:
                # Apply reducer to the model
                reduced_models.append(reducers[zone].reduce(model))
            else:
                # If no reducer is specified directly transfer the model
                reduced_models.append(model)

        return ModelReducer(reduced_models, self.zones, self.F, self.G, self.H, self.L, self.model_type)            


    def interconnect(self, u=None, y=None, component_label:str=None):
        """Interconnect all component models"""
        if u is None:
            u = lambda u: u[u.type == "device"]
        if y is None:
            y = lambda y:y

        return type_to_class[self.model_type].from_interconnected(self.models, self.connections, u, y, component_label)