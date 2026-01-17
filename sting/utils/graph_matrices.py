import numpy as np
import networkx as nx
from scipy.linalg import block_diag

from sting.system.selections import find_tagged

# from sting.models import ComponentConnections


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
            i, j = row_tuple.from_bus_id, row_tuple.to_bus_id

            # Calculate branch admittance
            z = complex(row_tuple.r_pu, row_tuple.l_pu)
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
            i = int(row_tuple.bus_id)

            # Calculate shunt admittance
            y_shunt = complex(row_tuple.g_pu, row_tuple.b_pu)

            # Add shunt admittance to the current admittance matrix
            Y[i, i] += y_shunt

    return Y

def build_admittance_matrix_from_lines(num_buses: int, lines: list):

    Y = np.zeros((num_buses, num_buses), dtype=complex)

    for line in lines:
        i = line.from_bus_id
        j = line.to_bus_id

        z = complex(line.r_pu, line.x_pu)
        y = 1.0 / z

        y_shunt = complex(line.g_pu, line.b_pu)

        Y[i, j] -= y
        Y[j, i] -= y

        Y[i, i] += y + y_shunt
        Y[j, j] += y + y_shunt

    return Y

def build_network_graph_from_lines(buses:list, lines: list, include_weights: bool = True):
    """
    Create a graph of the system. 

    TODO: This should probably be an attribute in the system.
          We will need to use a MultiGraph to support parallel edges.
          This could probably be the basis for a new class with methods like
            - build admittance matrix
            - build incidence matrix 
            - etc. 

          We should probably only save (type, id, name) to keep the
          graph lightweight (avoid redundant info). 

    DEPENDENCIES: Kron reduction module.
    """
    G = nx.Graph()
    G.add_nodes_from([bus.name for bus in buses])
    G.add_edges_from([(line.from_bus, line.to_bus) for line in lines])

    if include_weights == True:
        nx.set_edge_attributes(G, 0, "cap_existing_power_MW")
        for line in lines:
            u, v = line.from_bus, line.to_bus
            w = line.cap_existing_power_MW

            # Combine parallel edges by summing weights
            G[u][v]["cap_existing_power_MW"] += w

    return G

def build_generation_connection_matrix(num_buses: int, gen_bus: list):

    num_gens = len(gen_bus)

    gen_cx = np.zeros((num_gens, num_buses))

    for k in range(num_gens):
        connection_bus = gen_bus[k]
        gen_cx[k, connection_bus] = 1

    return gen_cx


def build_oriented_incidence_matrix(num_buses: int, branch_data: list):

    num_branches = len(branch_data)
    or_inc = np.zeros((num_buses, num_branches))

    for k in range(num_branches):
        from_bus = branch_data[k][0]
        to_bus = branch_data[k][1]

        or_inc[from_bus, k] = -1
        or_inc[to_bus, k] = +1

    return or_inc


def get_ccm_matrices(system, attribute, dimI):
    """ """
    gen_buses, gen_ssm = system.generators.select("bus_id", attribute)
    from_bus, to_bus, br_ssm = system.branches.select("from_bus_id", "to_bus_id", attribute)

    sh_ssm, = system.shunts.select(attribute)

    # Get the number of buses of the full system
    n_buses = len(system.bus)

    # Build generation connection matrix
    gen_cx = build_generation_connection_matrix(n_buses, gen_buses)

    # List containing the tuples (from_bus, to_bus) of the branches
    br_from_to = list(zip(from_bus, to_bus))
    # Build oriented incidence matrix
    or_inc = build_oriented_incidence_matrix(n_buses, br_from_to)

    # Build unoriented incidence matrix
    un_inc = abs(or_inc)

    d_gen, g_gen, y_gen = 0, 0, 0
    for ssm in gen_ssm:
        d_gen += ssm.u.n_device  # number of generator device-side inputs
        g_gen += ssm.u.n_grid  # number of generator grid-side inputs
        y_gen += len(ssm.y)  # number of generator outputs

    g_br, y_br = 0, 0
    for ssm in br_ssm:
        g_br += ssm.u.n_grid  # number of branch grid-side inputs
        y_br += len(ssm.y)  # number of branch outputs

    g_sh, y_sh = 0, 0
    for ssm in sh_ssm:
        g_sh += ssm.u.n_grid  # number of shunt grid-side inputs
        y_sh += len(ssm.y)  # number of shunt outputs

    y = y_gen + y_sh + y_br  # number of system outputs

    # Construct matrix F. We first build its blocks.
    F11 = np.zeros((d_gen, y_gen))
    F12 = np.zeros((d_gen, y_sh))
    F13 = np.zeros((d_gen, y_br))

    F21 = np.zeros((g_gen, y_gen))
    F22 = np.kron(gen_cx, np.eye(dimI))
    F23 = np.zeros((g_gen, y_br))

    F31 = np.kron(np.transpose(gen_cx), np.eye(dimI))
    F32 = np.zeros((g_sh, y_sh))
    F33 = np.kron(or_inc, np.eye(dimI))

    F41 = np.zeros((g_br, y_gen))
    # fmt: off
    F42 = np.kron( 0.5*(np.kron( np.transpose(un_inc) , np.array([[1], [1]]) )
                        + np.kron( np.transpose(or_inc) , np.array([[-1], [1]]) ) ) 
                         , np.eye(dimI) )
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
    # fmt: on
    # Construct matrix H and L
    H = np.eye(y)
    L = np.zeros((y, d_gen))

    return F, G, H, L


def build_ccm_permutation(system):
    """
    Build the permutation matrices from Lemma 1 and 2.
    """
    # Create empty lists for transformations, list order follows that of generator_types_list
    Y1, Y2, T1 = [], [], []
    # Iterate over the all generator types: [inf_src, gfmi_a, gfmi_b, ...]
    generator_types = find_tagged(system, "generator")
    for gen_type in generator_types:
        # Number of generators of the given type
        gens = getattr(system, gen_type)
        n = len(gens)

        if n == 0:
            continue

        # Note: all generators in 'gens' of the same class and will have
        # the same inputs and outputs. Thus, we only need to examine gen_0.
        d = gens[0].ssm.u.n_device  # number of device-side inputs
        g = gens[0].ssm.u.n_grid  # number of grid-side inputs

        # Build transformation (permutation) matrices
        X1 = np.kron(np.eye(n), np.hstack((np.eye(d), np.zeros((d, g)))))
        X2 = np.kron(np.eye(n), np.hstack((np.zeros((g, d)), np.eye(g))))
        # Note: T1, and T2 are permutation matrices, thus inverse == transpose
        T1.append(np.linalg.inv(np.vstack((X1, X2))))

        # Also, append transformations that are used later
        Y1.append(np.hstack((np.eye(n * d), np.zeros((n * d, n * g)))))
        Y2.append(np.hstack((np.zeros((n * g, n * d)), np.eye(n * g))))

    T1 = block_diag(*T1)
    # Build transformations
    Y1 = block_diag(*Y1)
    Y2 = block_diag(*Y2)
    T2 = np.linalg.inv(np.vstack((Y1, Y2)))

    return T1 @ T2
