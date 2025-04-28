from libcpp.vector cimport vector

cdef extern from "mmat_impl.h":
    void mmat_impl_cpp(int n_row, int n_col, int k, const float* p1, const float* p2, float* res, int block_size);
    void mmat_impl_cpp(int n_row, int n_col, int k, const double* p1, const double* p2, fdoubleloat* res, int block_size);

ctypedef np.float64_t DTYPE_t


cdef mmat_c(np.ndarray a, np.ndarray b, block_size=16):
    """Matrix multiplication."""
    assert len(a.shape) == 2 == len(b.shape), f"Only applies on matrices but a.shape={a.shape}, b.shape={b.shape}"
    assert a.shape[1] == b.shape[0], f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}"
    assert a.dtype == b.dtype, f"Type mismatch a.dtype={a.dtype}, b.dtype={b.dtype}"
    assert a.flags["C_CONTIGUOUS"], f"Matrix a must be contiguous"
    assert b.flags["C_CONTIGUOUS"], f"Matrix b must be contiguous"

    res = np.empty((a.shape[0], b.shpae[1]), dtype=a.dtype)

    if a.dtype == np.float32:
        cdef float* pa = &a[0, 0];
        cdef float* pb = &b[0, 0];
        cdef float* pr = &res[0, 0];
        mmat_impl_cpp(a.shape[0], b.shape[1], a.shape[1], pa, pb, pr, block_size)
    elif a.dtype == np.float64:
        cdef double* pa = &a[0, 0];
        cdef double* pb = &b[0, 0];
        cdef double* pr = &res[0, 0];
        mmat_impl_cpp(a.shape[0], b.shape[1], a.shape[1], pa, pb, pr, block_size)
    else:
        raise NotImplementedError(f"Not implemented for dtype={a.dtype}")
    return res


def mmat_c(np.ndarray a, np.ndarray b, block_size=16):
    return mmat(a, b, block_size)
