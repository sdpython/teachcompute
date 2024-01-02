#
# module: teachcompute.validation.cython.experiment_cython
#
message(STATUS "+ CYTHON teachcompute.validation.cython.experiment_cython")

cython_add_module(
  experiment_cython
  ../teachcompute/validation/cython/experiment_cython.pyx
  OpenMP::OpenMP_CXX
  ../teachcompute/validation/cython/experiment_cython_.cpp)

#
# module: teachcompute.validation.cython.mul_cython_omp
#
message(STATUS "+ CYTHON teachcompute.validation.cython.mul_cython_omp")

cython_add_module(
  mul_cython_omp
  ../teachcompute/validation/cython/mul_cython_omp.pyx
  OpenMP::OpenMP_CXX
  ../teachcompute/validation/cython/mul_cython_omp_.cpp)

#
# module: teachcompute.validation.cython.td_mul_cython
#
message(STATUS "+ CYTHON teachcompute.validation.cython.td_mul_cython")

cython_add_module(
  td_mul_cython
  ../teachcompute/validation/cython/td_mul_cython.pyx
  OpenMP::OpenMP_CXX
  ../teachcompute/validation/cython/td_mul_cython_.cpp)
