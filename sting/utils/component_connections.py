import numpy as np
from scipy.linalg import block_diag

from sting.utils.graph_matrices import (
    build_generation_connection_matrix,
    build_oriented_incidence_matrix,
)


def get_ccm_matrices(system, attribute: str, dimI: int):
    """ 
    Returns the matrices the interconnection matrices 
    of a power system for Component Connection Method (CCM).
    
    Parameters
    ----------
    system: A STING power system instance.
    attribute: Model attribute to build CCM matrices for---e.g., "ssm", "qbm", "emt"
    dimI: If dq = 2, if abc = 3

    Returns
    -------
    Interconnection matrices F,G,H, and L such that:
        u_stack = F * y_stack + G * u_sys
        y_sys   = H * y_stack + L * u_sys 
    """
    # Select the bus id and small-signal state-space model (SSM) of all generators
    gen_buses, gen_models = system.ccm_generators.filter(lambda x: getattr(x, attribute) is not None).select("bus_id", attribute)
    # Select the from/to bus id of all branches and the SSM 
    from_bus, to_bus, br_models = system.ccm_branches.select("from_bus_id", "to_bus_id", attribute)
    # Select the SSMs of all shunts
    sh_models, = system.ccm_shunts.select(attribute)

    # Get the number of buses of the full system
    n_buses = len(system.buses)

    # Build generation connection matrix
    A_connection = build_generation_connection_matrix(n_buses, gen_buses)

    # List containing the tuples (from_bus, to_bus) of the branches
    br_from_to = list(zip(from_bus, to_bus))
    # Build oriented incidence matrix
    B_incidence = build_oriented_incidence_matrix(n_buses, br_from_to)

    d_gen, g_gen, y_gen = 0, 0, 0
    for model in gen_models:
        d_gen += model.u.n_device # number of generator device-side inputs
        g_gen += model.u.n_grid   # number of generator grid-side inputs
        y_gen += len(model.y)     # number of generator outputs

    d_br, g_br, y_br = 0, 0, 0
    for model in br_models:
        d_br += model.u.n_device  # number of branch device-side inputs
        g_br += model.u.n_grid    # number of branch grid-side inputs
        y_br += len(model.y)      # number of branch outputs

    d_sh, g_sh, y_sh = 0, 0, 0
    for model in sh_models:
        d_sh += model.u.n_device  # number of shunt device-side inputs
        g_sh += model.u.n_grid    # number of shunt grid-side inputs
        y_sh += len(model.y)      # number of shunt outputs

    y = y_gen + y_sh + y_br  # number of system-level outputs
    u = d_gen + d_sh + d_br  # number of system-level inputs

    # --------------------------------------------------
    # Construct matrix F
    # --------------------------------------------------
    # [ u_gen ]   [       ]   [ i_gen ]
    # [ u_sh  ] = [   F   ] * [ v_sh  ]
    # [ u_br  ]   [       ]   [ i_br  ]

    # d_gen = 0 * i_gen + 0 * v_sh + 0 * i_br
    F11 = np.zeros((d_gen, y_gen))
    F12 = np.zeros((d_gen, y_sh))
    F13 = np.zeros((d_gen, y_br))

    # g_gen = 0 * i_gen + (A \otimes I) * v_sh + 0 * i_br
    F21 = np.zeros((g_gen, y_gen))
    F22 = np.kron(A_connection, np.eye(dimI))
    F23 = np.zeros((g_gen, y_br))

    # d_sh = 0 * i_gen + 0 * v_sh + 0 * i_br
    F31 = np.zeros((d_sh, y_gen))
    F32 = np.zeros((d_sh, y_sh))
    F33 = np.zeros((d_sh, y_br))

    # g_sh =  (Aᵀ \otimes I) * i_gen + 0 * v_sh + (B \otimes I) * i_br
    F41 = np.kron(A_connection.T, np.eye(dimI))
    F42 = np.zeros((g_sh, y_sh))
    F43 = np.kron(B_incidence, np.eye(dimI))

    # d_br = 0 * i_gen + 0 * v_sh + 0 * i_br
    F51 = np.zeros((d_br, y_gen))
    F52 = np.zeros((d_br, y_sh))
    F53 = np.zeros((d_br, y_br))

    # g_br = 0 * i_gen + (0.5 *[(|B| \otimes [1 1]ᵀ) + (Bᵀ \otimes [-1 1]ᵀ)] \otimes I) * v_sh + 0 * i_br
    F61 = np.zeros((g_br, y_gen))
    F62 = np.kron( 
        0.5*(np.kron( abs(B_incidence.T) , np.array([[1], [1]]) )
        + np.kron( B_incidence.T , np.array([[-1], [1]]) ) ), 
        np.eye(dimI) )
    F63 = np.zeros( (g_br, y_br) )
    
    F = np.block([
        [F11, F12, F13],
        [F21, F22, F23],
        [F31, F32, F33],
        [F41, F42, F43],
        [F51, F52, F53],
        [F61, F62, F63],
    ])

    # --------------------------------------------------
    # Construct matrix G
    # --------------------------------------------------
    # [ u_gen ]   [       ]   [ d_gen ]
    # [ u_sh  ] = [   G   ] * [ d_sh  ]
    # [ u_br  ]   [       ]   [ d_br  ]

    G11 = np.hstack([np.eye(d_gen), np.zeros((d_gen, d_sh)), np.zeros((d_gen, d_br))])
    G21 = np.zeros((g_gen, u))
    G31 = np.hstack([np.zeros((d_sh, d_gen)), np.eye(d_sh), np.zeros((d_sh, d_br))])
    G41 = np.zeros((g_sh, u))
    G51 = np.hstack([np.zeros((d_br, d_gen)), np.zeros((d_br, d_sh)), np.eye(d_br)])
    G61 = np.zeros((g_br, u))

    G = np.block([
        [G11],[G21],[G31],[G41],[G51],[G61]
        ])
    
    # Construct matrix H and L
    H = np.eye(y)
    L = np.zeros((y, u))

    return F, G, H, L


def build_ccm_permutation(system, attribute:str, tag:str):
    """
    Build the permutation matrices from so that the grid level
    interconnection matrices from `get_ccm_matrices` will work
    directly (without the need to permute the device and grid
    side inputs of generators).

    Parameters
    ----------
    system: A STING power system instance.
    attribute: Model attribute to build CCM matrices for---e.g., "ssm", "qbm".

    Returns
    -------
    Permutation matrix T such that 
        F_new = T * F
        G_new = T * G
    """
    # Create empty lists for transformations, list order follows that of generator_types_list
    Y1, Y2, T1 = [], [], []
    # Iterate over the all generator types: [inf_src, gfmi_a, gfmi_b, ...]
    generator_types = system.find_tagged(tag)
    for gen_type in generator_types:
        # Number of generators of the given type
        gens = getattr(system, gen_type)
        n = len(gens)

        if ((n == 0) or (getattr(gens[0], attribute) is None)):
            continue

        # Note: all generators in 'gens' of the same class and will have
        # the same inputs and outputs. Thus, we only need to examine gen_0.
        d = getattr(gens[0], attribute).u.n_device  # number of device-side inputs
        g = getattr(gens[0], attribute).u.n_grid  # number of grid-side inputs

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
