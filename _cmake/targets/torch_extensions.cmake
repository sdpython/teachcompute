#
# module: teachcompute.torch_extensions.piecewise_linear_c
# site-packages/torch/include/torch/extension.h
#
if(CUDA_AVAILABLE)

  message(STATUS "+ teachcompute.torch_extension.piecewise_linear_c (CUDA)")

  find_package(Torch REQUIRED)

  message(STATUS "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")

  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
  get_target_property(TORCH_INTERFACE_LIB torch INTERFACE_LINK_LIBRARIES)
  string(REPLACE "/usr/local/cuda" ${CUDA_TOOLKIT_ROOT_DIR} TORCH_INTERFACE_LIB "${TORCH_INTERFACE_LIB}")
  set_target_properties(torch PROPERTIES INTERFACE_LINK_LIBRARIES ${TORCH_INTERFACE_LIB})

  cuda_pybind11_add_module(
    piecewise_linear_c
    ../teachcompute/torch_extensions/piecewise_linear_c.cpp)

  target_include_directories(
    piecewise_linear_c
    PRIVATE
    ${TORCH_INCLUDE_DIRS})
  target_link_libraries(piecewise_linear_c PRIVATE ${TORCH_LIBRARIES})

else()

  message(STATUS "+ teachcompute.torch_extension.piecewise_linear_c (CPU)")

  # find_package(Torch REQUIRED)

  message(STATUS "TORCH_INCLUDE=${TORCH_INCLUDE}")

  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

  cuda_pybind11_add_module(
    piecewise_linear_c
    ../teachcompute/torch_extensions/piecewise_linear_c.cpp)

  target_include_directories(
    piecewise_linear_c
    PRIVATE
    ${TORCH_INCLUDE_DIRS})
  target_link_libraries(piecewise_linear_c PRIVATE ${TORCH_INCLUDE})

endif()
