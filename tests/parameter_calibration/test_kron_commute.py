from sting.utils.dynamical_systems import kronecker_commute, reshape_transpose
import numpy as np

# Create the random number generator instance
rng = np.random.default_rng()

# 1. Single random integer between 0 and 9
for _ in range(100):
    n = rng.integers(low=1, high=100) 
    m = rng.integers(low=1, high=100) 
    A = kronecker_commute(n, m)
    B = reshape_transpose(n, m).todense()

    assert np.all(A == B)