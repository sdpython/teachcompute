#
# module: teachcompute.validation.cython.dot_blas_lapack
#
message(STATUS "+ CYTHON teachcompute.validation.cython.dot_blas_lapack")

cython_add_module(
  dot_blas_lapack
  ../teachcompute/validation/cython/dot_blas_lapack.pyx
  OpenMP::OpenMP_CXX)
