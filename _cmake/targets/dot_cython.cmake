#
# module: teachcompute.validation.cython.dot_cython
#
message(STATUS "+ CYTHON teachcompute.validation.cython.dot_cython")

cython_add_module(
  dot_cython
  ../teachcompute/validation/cython/dot_cython.pyx
  OpenMP::OpenMP_CXX
  ../teachcompute/validation/cython/dot_cython_.cpp)
