import numpy as np
import networkx as nx
from scipy.linalg import block_diag

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
            z = complex(row_tuple.r_pu, row_tuple.x_pu)
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


