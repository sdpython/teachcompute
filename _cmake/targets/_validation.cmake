#
# module: teachcompute.validation.cpu._validation
#
message(STATUS "+ PYBIND11 teachcompute.validation.cpu._validation")

add_library(lib_validation_cpp STATIC
  ../teachcompute/validation/cpu/vector_function.cpp)
target_compile_definitions(lib_validation_cpp PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_include_directories(lib_validation_cpp PRIVATE "${ROOT_INCLUDE_PATH}")
set_property(TARGET lib_validation_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)

local_pybind11_add_module(
  _validation OpenMP::OpenMP_CXX
  ../teachcompute/validation/cpu/_validation.cpp
  ../teachcompute/validation/cpu/thread_sum.cpp
  ../teachcompute/validation/cpu/vector_sum.cpp)
message(STATUS "    LINK _validation <- lib_validation_cpp")
target_include_directories(_validation PRIVATE "${ROOT_INCLUDE_PATH}")
target_link_libraries(_validation PRIVATE lib_validation_cpp common)

add_executable(
  test_validation_cpp
  ../_unittests/ut_validation/test_vector_sum.cpp)
target_compile_definitions(test_validation_cpp PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_include_directories(
  test_validation_cpp
  PRIVATE
  "${ROOT_PROJECT_PATH}"
  "${ROOT_INCLUDE_PATH}"
  "${ROOT_UNITTEST_PATH}")
message(STATUS "    LINK test_validation_cpp <- lib_validation_cpp")
target_link_libraries(
  test_validation_cpp
  PRIVATE
  lib_validation_cpp
  common)
add_test(NAME test_validation_cpp COMMAND test_validation_cpp)
