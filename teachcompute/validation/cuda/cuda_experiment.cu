#include "cuda_example.cuh"
#include "cuda_nvtx.cuh"
#include "cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://github.com/zchee/cuda-sample/blob/master/6_Advanced/reduction/reduction_kernel.cu

namespace cuda_example {

template <typename T> __device__ __forceinline__ T _add(const T a, const T b) {
  return a + b;
}

template <> __device__ __forceinline__ half _add(const half a, const half b) {
#if __CUDA_ARCH__ < 700
  return __float2half(__half2float(a) + __half2float(b));
#else
  return a + b;
#endif
}

template <>
__device__ __forceinline__ half2 _add(const half2 a, const half2 b) {
#if __CUDA_ARCH__ < 700
  return half2(__float2half(__half2float(a.x) + __half2float(b.x)),
               __float2half(__half2float(a.y) + __half2float(b.y)));
#else
  return a + b;
#endif
}

__global__ void vector_add_half(const half *a, const half *b, half *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = _add(a[i], b[i]);
  }
}

__global__ void vector_add_half2(const half2 *a, const half2 *b, half2 *c,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = _add(a[i], b[i]);
  }
}

std::tuple<double, double>
measure_vector_add_half(unsigned int size, const short *ptr1, const short *ptr2,
                        short *ptr3, int cudaDevice, int repeat) {
  if (sizeof(short) != sizeof(half)) {
    throw std::runtime_error("sizeof(short) != sizeof(half)");
  }
  NVTX_SCOPE("vector_add")

  checkCudaErrors(cudaSetDevice(cudaDevice));
  half *gpu_ptr1, *gpu_ptr2, *gpu_res;
  checkCudaErrors(cudaMalloc(&gpu_ptr1, size * sizeof(half)));
  checkCudaErrors(cudaMalloc(&gpu_ptr2, size * sizeof(half)));
  checkCudaErrors(cudaMalloc(&gpu_res, size * sizeof(half)));

  checkCudaErrors(
      cudaMemcpy(gpu_ptr1, ptr1, size * sizeof(half), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(gpu_ptr2, ptr2, size * sizeof(half), cudaMemcpyHostToDevice));

  constexpr int blockSize = 256;
  std::tuple<double, double> res;

  auto time0 = std::chrono::high_resolution_clock::now();

  // execute the code
  {
    int numBlocks = (size + blockSize - 1) / blockSize;
    NVTX_SCOPE("vector_add_half")
    for (int r = 0; r < repeat; ++r) {
      vector_add_half<<<numBlocks, blockSize>>>(
          reinterpret_cast<const half *>(gpu_ptr1),
          reinterpret_cast<const half *>(gpu_ptr2),
          reinterpret_cast<half *>(gpu_res), size);
    }
  }

  std::get<0>(res) = std::chrono::duration<double>(
                         std::chrono::high_resolution_clock::now() - time0)
                         .count();

  // execute the code

  time0 = std::chrono::high_resolution_clock::now();

  {
    int numBlocks = (size / 2 + blockSize - 1) / blockSize;
    NVTX_SCOPE("vector_add_half2")
    for (int r = 0; r < repeat; ++r) {
      vector_add_half2<<<numBlocks, blockSize>>>(
          reinterpret_cast<const half2 *>(gpu_ptr1),
          reinterpret_cast<const half2 *>(gpu_ptr2),
          reinterpret_cast<half2 *>(gpu_res), size / 2);
    }
  }

  std::get<1>(res) = std::chrono::duration<double>(
                         std::chrono::high_resolution_clock::now() - time0)
                         .count();

  checkCudaErrors(
      cudaMemcpy(ptr3, gpu_res, size * sizeof(half), cudaMemcpyDeviceToHost));

  // free the allocated vectors
  checkCudaErrors(cudaFree(gpu_ptr1));
  checkCudaErrors(cudaFree(gpu_ptr2));
  checkCudaErrors(cudaFree(gpu_res));

  return res;
}

} // namespace cuda_example
