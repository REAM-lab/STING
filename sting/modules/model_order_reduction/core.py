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
        self.ssm.sort_components(by="zone")
        # TODO: For each zone create a zone component (also need to implement reduced/grouped component class)
        # 

        # self.ssm.group_by("zone").stack()
        # self.ssm.connect_subsystems()


    def connect_subsystems(self):
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
        pass