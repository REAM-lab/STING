from sting.utils.graph_matrices import build_generation_connection_matrix, build_oriented_incidence_matrix
import numpy as np
from scipy.linalg import block_diag

from sting.system.core import System

def get_ccm_matrices(system: System, attribute: str, dimI: int):
    """ """
    gen_buses, gen_ssm = system.gens.select("bus_id", attribute)
    from_bus, to_bus, br_ssm = system.branches.select("from_bus_id", "to_bus_id", attribute)

    sh_ssm, = system.shunts.select(attribute)

    # Get the number of buses of the full system
    n_buses = len(system.buses)

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


def build_ccm_permutation(system: System):
    """
    Build the permutation matrices from Lemma 1 and 2.
    """
    # Create empty lists for transformations, list order follows that of generator_types_list
    Y1, Y2, T1 = [], [], []
    # Iterate over the all generator types: [inf_src, gfmi_a, gfmi_b, ...]
    generator_types = system.find_tagged("generator")
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
