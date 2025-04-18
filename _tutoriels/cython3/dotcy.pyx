# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

ctypedef np.float64_t DTYPE_t


@boundscheck(False)
@wraparound(False)
def add_scaled(np.ndarray[DTYPE_t, ndim=1] a not None,
               np.ndarray[DTYPE_t, ndim=1] b not None):
    cdef int n = a.shape[0]
    assert b.shape[0] == n

    cdef DTYPE_t[::1] a_view = a
    cdef DTYPE_t[::1] b_view = b
    cdef DTYPE_t[::1] out = np.empty(n, dtype=np.float64)

    cdef int i
    with nogil:
        for i in range(n):
            out[i] = a_view[i] + 2.0 * b_view[i]

    return np.asarray(out)
