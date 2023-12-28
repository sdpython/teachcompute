#
# module: teachcompute.validation.cuda.cuda_monitor
#
if(CUDA_AVAILABLE)

  message(STATUS "+ PYBIND11 CUDA teachcompute.validation.cuda.cuda_monitor")

  cuda_pybind11_add_module(
    cuda_monitor
    ../teachcompute/validation/cuda/cuda_monitor.cpp)

  target_include_directories(cuda_monitor PRIVATE ${ROOT_INCLUDE_PATH})
  target_link_libraries(cuda_monitor PRIVATE common CUDA::nvml)

endif()
