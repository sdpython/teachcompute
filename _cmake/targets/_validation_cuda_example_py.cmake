#
# module: teachcompute.validation.cuda.cuda_example_py
#
if(CUDA_AVAILABLE)

  message(STATUS "+ PYBIND11 CUDA teachcompute.validation.cuda.cuda_example_py")

  cuda_pybind11_add_module(
    cuda_example_py
    ../teachcompute/validation/cuda/cuda_example_py.cpp
    ../teachcompute/validation/cuda/cuda_tensor.cu
    ../teachcompute/validation/cuda/cuda_example.cu
    ../teachcompute/validation/cuda/cuda_example_reduce.cu
    ../teachcompute/validation/cuda/cuda_experiment.cu)

  target_include_directories(cuda_example_py PRIVATE ${ROOT_INCLUDE_PATH})
  target_link_libraries(cuda_example_py PRIVATE common)

endif()
