cmake_minimum_required(VERSION 3.25.0)
project(teachcompute VERSION ${teachcompute_VERSION})


# Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
# see https://cmake.org/cmake/help/latest/policy/CMP0104.html
# cmake_policy(SET CMP0104 OLD) # deprecated

list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/externals")


#
# Packages and constants
#

include("constants.cmake")
include("load_externals.cmake")

#
# modules
#

message(STATUS "--------------------------------------------")
set(ROOT_PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ROOT_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../teachcompute/cpp/include)
set(ROOT_UNITTEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../_unittests)
message(STATUS "ROOT_PROJECT_PATH=${ROOT_PROJECT_PATH}")
message(STATUS "ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}")
message(STATUS "ROOT_UNITTEST_PATH=${ROOT_UNITTEST_PATH}")
message(STATUS "--------------------------------------------")

#
# common libraries for everything
#

include("targets/common.cmake")

#
# standalone modules
#

include("targets/_validation.cmake")
include("targets/_validation_cuda_example_py.cmake")
include("targets/_validation_cuda_gemm.cmake")
include("targets/_validation_cuda_monitor.cmake")
include("targets/vector_function_cy.cmake")
include("targets/direct_blas_lapack_cy.cmake")
include("targets/dot_blas_lapack.cmake")
include("targets/dot_cython.cmake")
include("targets/experiment_cython.cmake")
include("targets/torch_extensions.cmake")

#
# finalize
#

include("finalize.cmake")

#
# write version
#

message(STATUS "---------------------------------------------------")
message(STATUS "CONTENT\n ${config_content}")
message(STATUS "WRITE _CONFIG: '${ROOT_PROJECT_PATH}/teachcompute/_config.py'")
file(WRITE "${ROOT_PROJECT_PATH}/teachcompute/_config.py" "${config_content}")
message(STATUS "WRITE _CONFIG: '${SETUP_BUILD_LIB}/teachcompute/_config.py'")
file(WRITE "${SETUP_BUILD_LIB}/teachcompute/_config.py" "${config_content}")

#
# test
#

include(CTest)
enable_testing()

#
# Final
#

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

#
# Final check
#

get_property(targets_list GLOBAL PROPERTY PACKAGES_FOUND)
message(STATUS "-------------------")
message(STATUS "CMAKE_PROJECT_NAME = ${CMAKE_PROJECT_NAME}")
message(STATUS "list of found packages")
foreach(target ${targets_list})
  message(STATUS "  ${target}")
endforeach()
message(STATUS "-------------------")
