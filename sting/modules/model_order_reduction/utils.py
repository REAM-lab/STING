import numpy as np
from scipy.linalg import solve, eig
from typing import Literal
from dataclasses import dataclass, field
from control import gram
from warnings import warn

from sting.utils.dynamical_systems import StateSpaceModel
from sting.modules.small_signal_modeling.core import SmallSignalModel
from sting.utils.matrix_tools import mat2cell
from sting.system.component import Component

def singular_perturbation(ss:StateSpaceModel, r:int) -> StateSpaceModel:
    """
    Return a reduced-order model by substituting the quasi-steady state
    model into the full-order dynamics.
    """
    n = ss.A.shape[0] # Order of the sstem
    split = [r, n-r]   # Number of states in each partition
    
    # Partition the state-space matrices
    A = mat2cell(ss.A, split, split)
    B = mat2cell(ss.B, split, None)
    C = mat2cell(ss.C, None, split)
    invA_11 = solve(A[1,1], np.eye(n-r))

    # Substituting QSS model into slow dynamics
    A_r = A[0,0] - A[0,1]@invA_11@A[1,0]
    B_r = B[0,0] - A[0,1]@invA_11@B[1,0]
    C_r = C[0,0] - C[0,1]@invA_11@A[1,0]
    D_r = ss.D - C[0,1]@invA_11@B[1,0]

    ss_r = StateSpaceModel(A=A_r, B=B_r, C=C_r, D=D_r)

    return ss_r
    

@dataclass(slots=True)
class BlockGramian:
    method: Literal["lyapunov", "structured"]
    type: Literal["controllability", "observability"]

    def solve(self, ssm:SmallSignalModel):

        # Compute a gramian using the specified method
        W = getattr(self, "_"+self.method)(ssm)
        
        # If W is NOT positive definite AND the smallest eigenvalue is 10^12
        # times smaller than the largest introduce a regularization term to W.
        eigenvalues = eig(W)
        min_ev = -min(eigenvalues)
        max_ev = max(eigenvalues)
        if (max_ev/min_ev >= 1e12) and (max_ev > 0):
            W = W + 2*min_ev*np.eye(W.size[0])

        
        start, stop = 0, 0

        # Assign block elements in W it each component
        for c in ssm.components:
            component = getattr(ssm.system, c.type)[c.id]
            n = component.ssm.A.size[0]
            stop += n

            if hasattr(component, "W_"+self.type[0]):
                W_i = W[start:stop, start:stop]
                gramian = getattr(component, "W_"+self.type[0])
                setattr(gramian, self.method, W_i)

            start += n

    def _lyapunov(self, ssm:SmallSignalModel) -> np.ndarray:
         # Compute the Gramian of the *system-level* state-space model
        sys = ssm.model.to_python_control()
        W = gram(sys=sys, type=self.type[0])
        return W

    def _structured(self, ssm:SmallSignalModel) -> np.ndarray:
        """
        TODO: Implement this LMI

        [A,B,C,D] = ssdata(sys_c);

        % If using observability gramian create the dual system
        if type == 'o'
            temp = {A', C', B'};
            [A, B, C] = temp{:};
        end

        % Build a block diagonal sdpvar
        X = cell(1, size(conponentModel));
        for k=1:size(conponentModel)
            X{k} = sdpvar(sys_order(k), sys_order(k), ...
                'symmetric');
        end
        X = blkdiag(X{:});

        % Minimize Trace(P) s.t. Lyapunov equation
        LMI = [X >= 0, A'*X+X*A + B*B' <= 0];
        % Assumes MOSEK optimizer is installed
        opt = sdpsettings('verbose', 1, 'solver', 'mosek');
        sol = optimize(LMI, [], opt);

        if sol.problem > 0
            error("Solver failed with value %g. " + ...
                "\n1 - The LMI is infeasible." + ...
                "\n2 - The LMI is unbounded.", sol.problem)
        end

        X = value(X);
        """
        pass

