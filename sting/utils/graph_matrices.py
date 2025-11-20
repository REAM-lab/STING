import numpy as np
from scipy.linalg import block_diag 
#from sting.models import ComponentConnections


def build_admittance_matrix(num_buses: int, branch_data=None, shunt_data=None):
    """
    Builds the bus admittance matrix (Y_bus) for a power system.

    Args:
        num_buses (int): The total number of buses in the system.
        branch_data (dataframe): It must include these columns:
                from_bus | to_bus | r | l
            0   

        shunt_data (list of tuple, optional): It must include these columns:
                bus | g | b
            0   
    Returns:
        Complex admittance matrix
        Real conductance matrix
        Real susceptance matrix
    """
    # Initialize an n x n matrix of zeros with complex data type, where n is the number of buses
    Y = np.zeros((num_buses, num_buses), dtype=complex)

    if branch_data is not None:
        # Calculate off-diagonal elements of the admitance matrix (Y_ij)
        for row_tuple in branch_data.itertuples():
            # As Python starts from zero, convert to 0-based indexing
            i, j = int(row_tuple.from_bus) - 1, int(row_tuple.to_bus) - 1
        
            # Calculate branch admittance
            z = complex(row_tuple.r, row_tuple.l)
            y = 1.0 / z
        
            # Calculate off-diagonal elements are the negative of the branch admittance
            Y[i, j] -= y
            Y[j, i] -= y

        # Calculate diagonal elements (Y_ii). Note: Y_ii = y_1i + y2i + ... + yii + ...
        for i in range(num_buses):
            # The diagonal element is the sum of all admittances connected to the bus
            # The negative before y_bus is because y_bus has considered negative above
            Y[i, i] = -np.sum(Y[i, :])
        
    # Add shunt admittances if provided
    if shunt_data is not None:
        for row_tuple in shunt_data.itertuples():
            # As Python starts from zero, convert to 0-based indexing
            i = int(row_tuple.bus_idx) - 1

            # Calculate shunt admittance 
            y_shunt = complex(row_tuple.g, row_tuple.b)

            # Add shunt admittance to the current admittance matrix
            Y[i, i] += y_shunt

          
    return Y

def build_generation_connection_matrix( num_buses: int, gen_bus: list):

    num_gens = len(gen_bus)

    gen_cx = np.zeros((num_gens, num_buses))
    gen_bus = np.array((gen_bus)).astype(int) # get a vector that contains the connection bus of all gens
    gen_bus -=1 

    for k in range(num_gens):
        gen_cx[k,gen_bus[k]] = 1

    return gen_cx

def build_oriented_incidence_matrix( num_buses: int, branch_data: list):

    num_branches = len(branch_data)
    or_inc = np.zeros((num_buses, num_branches))
    branch_data = np.array(branch_data).astype(int)
    branch_data -= 1

    for k in range(num_branches):
        or_inc[branch_data[k][0],k] = -1
        or_inc[branch_data[k][1],k] = +1

    return or_inc


def get_ccm_matrices(system):
        """
        
        """
        attrs = ["bus_idx", "ssm"]
        gen_buses, gen_ssm = system.view("generators", attrs=attrs)
        
        attrs = ["from_bus", "to_bus", "ssm"]
        from_bus, to_bus, br_ssm = system.view("branches", attrs=attrs)
        
        sh_ssm = system.view("shunts", attrs="ssm")
        
        # Get the number of buses of the full system
        n_buses = system.length(view="buses")

        # Build generation connection matrix
        gen_cx = build_generation_connection_matrix(n_buses, list(gen_buses))
        
        
        # List containing the tuples (from_bus, to_bus) of the branches
        br_from_to =   list(zip(from_bus, to_bus))
        # Build oriented incidence matrix
        or_inc = build_oriented_incidence_matrix(n_buses, br_from_to)

        # Build unoriented incidence matrix
        un_inc = abs(or_inc)
        
        
        d_gen, g_gen, y_gen = 0, 0, 0
        for ssm in gen_ssm:
            d_gen += ssm.u.n_device # number of generator device-side inputs 
            g_gen += ssm.u.n_grid   # number of generator grid-side inputs 
            y_gen += len(ssm.y)     # number of generator outputs
            
        g_br, y_br = 0, 0
        for ssm in br_ssm:
            g_br += ssm.u.n_grid    # number of branch grid-side inputs 
            y_br += len(ssm.y)      # number of branch outputs
        
        g_sh, y_sh = 0, 0
        for ssm in sh_ssm:
            g_sh += ssm.u.n_grid    # number of shunt grid-side inputs 
            y_sh += len(ssm.y)      # number of shunt outputs
        
        y = y_gen + y_sh + y_br     # number of system outputs

        # Construct matrix F. We first build its blocks.
        F11 = np.zeros( (d_gen, y_gen) )
        F12 = np.zeros( (d_gen, y_sh) )
        F13 = np.zeros( (d_gen, y_br) )

        F21 = np.zeros( (g_gen, y_gen) )
        F22 = np.kron( gen_cx, np.eye(2) )
        F23 = np.zeros( (g_gen, y_br) )

        F31 = np.kron( np.transpose(gen_cx), np.eye(2))
        F32 = np.zeros( (g_sh, y_sh) )
        F33 = np.kron( or_inc, np.eye(2) )

        F41 = np.zeros( (g_br, y_gen) )
        F42 = np.kron( 0.5*(np.kron( np.transpose(un_inc) , np.array([[1], [1]]) )
                            + np.kron( np.transpose(or_inc) , np.array([[-1], [1]]) ) ) 
                             , np.eye(2) )
        F43 = np.zeros( (g_br, y_br) )

        F = np.block( [[F11, F12, F13],
                       [F21, F22, F23],
                       [F31, F32, F33],
                       [F41, F42, F43] ])
        
        # Construct matrix G. We first build its blocks.
        G11 = np.eye(d_gen)
        G21 = np.zeros((g_gen, d_gen))
        G31 = np.zeros((g_sh, d_gen))
        G41 = np.zeros((g_br, d_gen))

        G = np.block([[G11], 
                      [G21],
                      [G31],
                      [G41]])
        
        # Construct matrix H and L
        H = np.eye(y)
        L = np.zeros((y, d_gen))
        
        T = build_ccm_permutation(system)
        T = block_diag(T, np.eye(F.shape[0] - T.shape[0]))

        F = T @ F
        G = T @ G

        return F, G, H, L


def build_ccm_permutation(system):
    """
    Build the permutation matrices from Lemma 1 and 2. 
    """
    # Create empty lists for transformations, list order follows that of generator_types_list
    Y1, Y2, T1 = [], [], []

    # Iterate over the all generator types: [inf_src, gfmi_a, gfmi_b, ...]
    for gen_type in system.views["generators"]:
        # Number of generators of the given type
        n = system.length(group=gen_type)

        if n == 0:
            continue
        
        # Note: all generators in 'gens' of the same class and will have 
        # the same inputs and outputs. Thus, we only need to examine gen_0.
        gen_0 = next(system.values(gen_type))

        d = gen_0.ssm.u.n_device # number of device-side inputs 
        g = gen_0.ssm.u.n_grid   # number of grid-side inputs 

        # Build transformation (permutation) matrices
        X1 = np.kron(np.eye(n), 
                            np.hstack( ( np.eye(d), np.zeros((d, g)) ) ) )
        X2 = np.kron(np.eye(n), 
                            np.hstack( ( np.zeros((g, d)), np.eye(g) ) ) )
        # Note: T1, and T2 are permutation matrices, thus inverse == transpose
        T1.append(np.linalg.inv(np.vstack((X1, X2))))

        # Also, append transformations that are used later
        Y1.append(np.hstack(( np.eye(n*d), np.zeros((n*d, n*g)) )) )
        Y2.append(np.hstack(( np.zeros((n*g, n*d)), np.eye(n*g) )) )

    T1 = block_diag(*T1)
    # Build transformations
    Y1 = block_diag(*Y1)
    Y2 = block_diag(*Y2)
    T2 = np.linalg.inv( np.vstack((Y1, Y2)) )

    return T1 @ T2