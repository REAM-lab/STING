import numpy as np
import pandas as pd
import collections as col


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