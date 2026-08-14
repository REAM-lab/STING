import numpy as np

b = 0.1
kd = 0.05

A = np.array([[0, 1, 0, 0],
              [-b, -kd, b, 0],
              [0, 0, 0, 1],
              [b, 0, -b, -kd]])

eig = np.linalg.eigvals(A)

print("Eigenvalues of the original system:")
for x in eig:
    print(x)

angle_positions = np.array([1, 0, 1, 0])  # Example angle positions for the states

reference = 1
reference = reference - 1  # Convert to zero-based index

# Indices of states with angle positions == 1
angle_idx = np.flatnonzero(angle_positions)

# Eliminate reference from one_indices
one_indices = np.delete(angle_idx, reference)

# Position of reference
reference_index = angle_idx[reference]

# create identity matrix
T = np.eye(A.shape[0])

# Vectorized assignment: set T[one_indices, reference_index] = -1
T[one_indices, reference_index] = -1

A_t = T @ A @ np.linalg.inv(T)

# Eliminate reference row and column
A_t_reduced = np.delete(A_t, reference_index, axis=0)  #
A_t_reduced = np.delete(A_t_reduced, reference_index, axis=1)

eig_reduced = np.linalg.eigvals(A_t_reduced)

print("\nEigenvalues of the reduced system:")
for x in eig_reduced:
    print(x)
