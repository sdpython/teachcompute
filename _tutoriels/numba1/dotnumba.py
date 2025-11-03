import numpy as np
from numba import njit, prange


@njit(parallel=True)
def add_scaled_parallel(a, b):
    out = np.empty(a.shape, dtype=a.dtype)
    for i in prange(a.shape[0]):
        out[i] = a[i] + 2.0 * b[i]
    return out


if __name__ == "__main__":
    t = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    print(add_scaled_parallel(t, t))
