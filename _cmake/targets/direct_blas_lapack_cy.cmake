#
# module: teachcompute.validation.cython.direct_blas_lapack_cy
#
message(STATUS "+ CYTHON teachcompute.validation.cython.direct_blas_lapack_cy")

cython_add_module(
  direct_blas_lapack_cy
  ../teachcompute/validation/cython/direct_blas_lapack_cy.pyx
  OpenMP::OpenMP_CXX)
