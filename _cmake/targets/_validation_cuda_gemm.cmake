#
# module: teachcompute.validation.cuda.cuda_gemm
#
if(CUDA_AVAILABLE)

  message(STATUS "+ PYBIND11 CUDA teachcompute.validation.cuda.gemm")

  cuda_pybind11_add_module(
    cuda_gemm
    ../teachcompute/validation/cuda/cuda_gemm.cpp
    ../teachcompute/validation/cuda/cuda_gemm.cu)

  target_include_directories(cuda_gemm PRIVATE ${ROOT_INCLUDE_PATH})
  target_link_libraries(cuda_gemm PRIVATE common)

endif()
