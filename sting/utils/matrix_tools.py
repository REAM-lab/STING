import numpy as np
import polars as pl

def matrix_to_csv(filepath: str, matrix: np.ndarray, index: list, columns: list):
    """Write a numpy matrix to a tabular CSV with labeled rows and columns"""
    # Cast index and columns to strings (if they are not strings already)
    index = [str(s) for s in index if not isinstance(s, str)]
    columns = [str(s) for s in columns if not isinstance(s, str)]

    df = (
        pl.DataFrame(matrix, schema=columns)
        .with_columns(pl.Series(name="Index", values=index))
        # Reorder columns to place the 'Index' first    
        .select(["Index"] + columns)
    )
    df.write_csv(filepath)


def csv_to_matrix(filepath):
    df = pl.read_csv(filepath)
    # Unpack matrix, index, and columns
    matrix = df.drop("Index").to_numpy()
    index = df["Index"].to_list()
    columns = list(df.columns)
    return matrix, index, columns


def mat2cell(A: np.ndarray, m: list, n: list) -> np.ndarray:
    """Python clone of MATLAB mat2cell"""
    if m is None:
        m = [A.shape[0]]
    if n is None:
        n = [A.shape[1]]

    # Create a list of lists to hold the sub-arrays
    cell_array = np.empty((len(m), len(n)), dtype=object)

    r_start = 0
    for i, r_len in enumerate(m):
        c_start = 0
        for j, c_len in enumerate(n):
            cell_array[i,j] = A[
                r_start: r_start + r_len, c_start : c_start + c_len
            ]
            c_start += c_len
        r_start += r_len

    return cell_array


def cell2mat(C):
    """Python clone of MATLAB cell2mat"""
    if len(C) == 1:
        return np.vstack(C)
    return np.vstack([np.hstack(row) for row in C])


def block_permute(X, rows, cols, index):
    # Returns a matrix Y whose block matrix entries (given by rows and
    # cols) have been re-ordered according to an index.
    X = mat2cell(X, rows, cols)
    Y = np.empty(X.shape, dtype=object)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Y_ij = X_nm where the index maps n/m to i/j
            n = index[i]
            m = index[j]
            Y[i, j] = X[n, m]
    return cell2mat(Y)