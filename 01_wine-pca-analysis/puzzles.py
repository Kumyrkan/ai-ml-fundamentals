import numpy as np


# ROTATE 90° CLOCKWISE

def rotate_90(m):
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")

    return np.rot90(m, k=-1)


#  TRANSPOSE 

def transpose_no_T(m):
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")

    
    return np.einsum("ij->ji", m)


#  MAGIC SQUARE CHECK 

def is_magic_square(m):
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return False

    n, mcols = m.shape
    if n != mcols:
        return False

    row_sums = m.sum(axis=1)
    col_sums = m.sum(axis=0)

    diag1 = np.trace(m)
    diag2 = np.trace(np.fliplr(m))

    target = row_sums[0]

    return (
        np.all(row_sums == target)
        and np.all(col_sums == target)
        and np.isclose(diag1, target)
        and np.isclose(diag2, target)
    )


# BLOCK TRACE (reshape-based vectorization)

def block_trace(m, k):
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        raise ValueError("Input must be a 2D numpy array")

    if m.shape[0] != m.shape[1]:
        raise ValueError("Matrix must be square")

    n = m.shape[0]

    if n % k != 0:
        raise ValueError("k must divide matrix size")

    if k == n:
        return np.array([[m[0, 0]]])

    blocks = m.reshape(n // k, k, n // k, k)
    return np.trace(blocks, axis1=1, axis2=3)


# TOP-K INDICES (argpartition optimized)

def top_k_indices(v, k):
    if not isinstance(v, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if v.ndim != 1:
        raise ValueError("Input must be a 1D array")

    if k <= 0 or k > len(v):
        raise ValueError("Invalid k")

    idx = np.argpartition(v, -k)[-k:]
    return idx[np.argsort(v[idx])[::-1]]