#
# module: teachcompute.validation.cython.vector_function_cy
#
message(STATUS "+ CYTHON teachcompute.validation.cython.vector_function_cy")

cython_add_module(
  vector_function_cy
  ../teachcompute/validation/cython/vector_function_cy.pyx
  OpenMP::OpenMP_CXX
  ../teachcompute/validation/cpu/vector_function.cpp)
