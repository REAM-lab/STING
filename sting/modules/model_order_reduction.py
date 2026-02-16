# ----------------------
# Import python packages
# ----------------------
from dataclasses import dataclass, field

# ------------------
# Import sting code
# ------------------
from sting.system.core import System
from sting.modules.small_signal_modeling import SmallSignalModel


@dataclass(slots=True)
class ReducedOrderModel:
    ssm: SmallSignalModel
    system: System

    def __post_init__(self):
        """Construct zone level models from the SmallSignalModel"""
        
        # Sort component level model by zone
        self.ssm.sort_components_by(by="zone")
        # TODO: For each zone create a zone component (also need to implement reduced/grouped component class)
        # 


    def connect_subsystems(self):
        """
        Now we have eliminated extra outputs from each subsystem but there are no 
        internal connections within subsystems. Here we remove all inputs and 
        outputs from each subsystem that have no contribution to y_C.
        
        Recall from CCM that
            u_B = F * y_B + G * u_C
            y_C = H * y_B + L * u_C
        
        Specifically, we construct two more maskings for u_B and y_B. These
        correspond to inputs and outputs that have no contribution to u_C or y_C,
        and thus can be eliminated from each subsystem without any effect on the 
        fully interconnected dynamics of G_C(s). We the matrix construct X to 
        mask u_B such that u_B(mask) = X * u_B meaning
            X * u_B = (X*F) * y_B + (X*G) * u_B
        and the matrix Y to mask y_B such that y_B(mask) = Y * y_B meaning
            u_B = (F*Y) * y_B + G * u_C
            y_C = (H*Y) * y_B + L * u_C
        """
    

    # Read zone file to determine which subsystems are reducible
    input.filename = 'inputs/zones.csv';
    zones = csv2ntst(input);
    s.reducible = [zones.reducible].';
    z = height(s) # Number of zones

    # Number of stacked/grid-side inputs and outputs in each zone
    y_stack = cell_size(s.y_stack, 1);
    u_stack = cell_size(s.u_stack, 1);

    # Block partition the CCM matrices
    F = mat2cell(c.F, u_stack, y_stack);
    G = mat2cell(c.G, u_stack, width(c.G));
    H = mat2cell(c.H, height(c.H), y_stack);
    Matrices used in LFT to connect components within each subsystem
    diagF = cell(z,1);
    X = cell(z,1);
    Y = cell(z,1);

    for i = 1:z
        All indices excluding i
        j = setdiff(1:z, i);
        Inputs in u_B with outputs from another subsystem
        [u_B, ~] = find_unique([G{i}, F{i, j}]);    
        Outputs in y_B with contributions from u_B or y_c
        [~, y_B] = find_unique([H{i}; vertcat(F{j, i})]);

        Update the entries in the subsystems table
        s.u_stack{i} = s.u_stack{i}(u_B,:);
        s.y_stack{i} = s.y_stack{i}(y_B,:);

        Connect all components within the subsystem
        m = length(u_B);
        n = length(y_B);
        X_i = full(sparse(1:m, u_B, ones(1,m), m, height(F{i,i})));
        Y_i = full(sparse(y_B, 1:n, ones(1,n), width(F{i,i}), n));
        G_i = LFT(S_i,, H_{i})
        S_i = [zeros(n,m), Y_i'; 
            X_i'      , F{i,i}];
        s.sys{i} = lft(S_i, s.sys{i});
        
        Save the matrices used in the LFT
        diagF{i} = F{i,i};
        X{i} = X_i;
        Y{i} = Y_i;
    end

    X = blkdiag(X{:});
    Y = blkdiag(Y{:});
    diagF = blkdiag(diagF{:});

    c.F = X*(c.F-diagF)*Y;
    c.G = X*c.G;
    c.H = c.H*Y;
    end



    """
    function [c, s] = eliminate_outputs(c, s)
    At this point we have simply grouped the components to be ordered by each
    subsystem. There are no internal connections within subsystems. For each
    reducible subsystem mask all grid-side outputs (y_C) that are not inputs 
    to another subsystem.
    %
    We constuct Z to mask y_C such that y_C(mask) = Z * y_C meaning
        Z * y_C = (Z*H) * y_B + (Z*L) * u_C

    Read zone file to determine which subsystems are reducible
    input.filename = 'inputs/zones.csv';
    zones = csv2ntst(input);
    s.reducible = [zones.reducible].';
    z = height(s); Number of zones

    Number of stacked/grid-side inputs and outputs in each zone
    y_stack = cell_size(s.y_stack, 1);
    u_stack = cell_size(s.u_stack, 1);
    Mask outputs and block partition the CCM matrices
    F = mat2cell(c.F, u_stack, y_stack);

    k = cumsum([0; y_stack]); Cumulative number of outputs for each subsystem
    y_C = cell(z,1);
    for i = 1:z    
        
        All indices excluding i
        j = setdiff(1:z, i);

        if s.reducible(i)
            Get outputs that are inputs to another subsystem, given by the 
            nonzero columns of F_{j,i} 
            [~, y_C{i}] = find_unique(vertcat(F{j, i}));
        else
            Otherwise select keep all outputs 
            y_C{i} = (1:width(F{i,i}))';
        end
        Make index "global"
        y_C{i} = y_C{i} + k(i);
        
        Update the entries in the subsystems table
        s.y_grid{i} = s.y_grid{i}(y_C{i} - k(i),:);
        s.y_id{i} = s.y_id{i}(y_C{i} - k(i));
    end
    Construct masking matrix for H and L
    mask = vertcat(y_C{:});
    m = length(mask);
    Z = full(sparse(1:m, mask, ones(1,m), m, k(z+1)));
    Mask outputs in H and L
    c.H = Z*c.H;
    c.L = Z*c.L;
    end    
    """