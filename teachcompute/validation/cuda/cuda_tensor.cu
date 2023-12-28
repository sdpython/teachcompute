#include "cuda_tensor.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>
#include "teachcompute_helpers.h"

namespace cuda_example {

int32_t type_size(cudaDataType_t element_type) {
  switch (element_type) {
  case CUDA_R_32I:
  case CUDA_R_32F:
    return 4;
  case CUDA_R_16F:
  case CUDA_R_16BF:
    return 2;
  case CUDA_R_8I:
  case CUDA_R_8U:
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
  case CUDA_R_8F_E4M3:
  case CUDA_R_8F_E5M2:
#endif
    return 1;
  default:
    NVTE_CHECK(false,
               teachcompute_helpers::MakeString("Unkown data type ", element_type,
                                                 " and this CUDA version ", CUDA_VERSION, "."));
  }
}

void TensorData::allocate(cudaDataType_t dtype, std::size_t size, TensorDevice device) {
  this->dtype = dtype;
  this->size = size;
  this->device = device;
  switch (device) {
  case TensorDevice::CPU:
    dptr = malloc(size * type_size(dtype));
    break;
  case TensorDevice::CUDA:
    if (cudaMalloc(&dptr, size * type_size(dtype)) != cudaSuccess) {
      NVTE_ERROR(
          teachcompute_helpers::MakeString("Unable to allocate ", size, " bytes on GPU."));
    }
    break;
  }
}

void TensorData::free() {
  if (dptr != nullptr) {
    switch (device) {
    case TensorDevice::CPU:
      ::free(dptr);
      break;
    case TensorDevice::CUDA:
      NVTE_CHECK_CUDA(cudaFree(dptr));
      break;
    }
    dptr = nullptr;
  }
}

void TensorData::copy_from_cpu(void *ptr) {
  switch (device) {
  case TensorDevice::CPU:
    memcpy(dptr, ptr, type_size(dtype) * size);
    break;
  case TensorDevice::CUDA:
    NVTE_CHECK_CUDA(cudaMemcpy(dptr, ptr, type_size(dtype) * size, cudaMemcpyHostToDevice));
    break;
  default:
    NVTE_CHECK(false, teachcompute_helpers::MakeString("Unsupported device ", (int)device,
                                                        " for copy_from_cpu."));
  }
}

Tensor::Tensor(const char *name, std::size_t size, cudaDataType_t dtype, TensorDevice device,
               TensorDevice scale_device) {
  this->name = name;
  data.allocate(dtype, size, device);
}

Tensor::~Tensor() {
  data.free();
  scale.free();
  scale_inv.free();
  amax.free();
}

__global__ void generateRandomFloat16(__half *randomFloat16, int numElements,
                                      unsigned int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numElements) {
    curandState state;
    curand_init(seed, tid, 0, &state);
    float randValue = curand_uniform(&state);
    randomFloat16[tid] = __float2half(randValue);
  }
}

__global__ void generateRandomBFloat16(__nv_bfloat16 *randomFloat16, int numElements,
                                       unsigned int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numElements) {
    curandState state;
    curand_init(seed, tid, 0, &state);
    float randValue = curand_uniform(&state);
    randomFloat16[tid] = __float2bfloat16(randValue);
  }
}

__global__ void generateRandomInt8x4(int *randomInt8, int numElements, unsigned int seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numElements / 4) {
    curandState state;
    curand_init(seed, tid, 0, &state);
    int randValue = curand_poisson(&state, 1);
    randomInt8[tid] = randValue;
  }
}

void Tensor::rnd() {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  switch (data.dtype) {
  case CUDA_R_32F:
    curandGenerateUniform(gen, static_cast<float *>(data.dptr), data.size);
    break;
  case CUDA_R_16F: {
    int blockSize = 256;
    int numBlocks = (data.size + blockSize - 1) / blockSize;
    generateRandomFloat16<<<numBlocks, blockSize>>>(static_cast<__half *>(data.dptr), data.size,
                                                    0);
    cudaDeviceSynchronize();
  } break;
  case CUDA_R_16BF: {
    int blockSize = 256;
    int numBlocks = (data.size + blockSize - 1) / blockSize;
    generateRandomBFloat16<<<numBlocks, blockSize>>>(static_cast<__nv_bfloat16 *>(data.dptr),
                                                     data.size, 0);
    cudaDeviceSynchronize();
  } break;
  case CUDA_R_8I: {
    int blockSize = 256;
    int numBlocks = (data.size + blockSize - 1) / blockSize;
    generateRandomInt8x4<<<numBlocks, blockSize>>>(static_cast<int *>(data.dptr), data.size, 0);
    cudaDeviceSynchronize();
  } break;
  default:
    NVTE_CHECK(false, teachcompute_helpers::MakeString("Unsupported dtype ", data.dtype,
                                                        " for rnd."));
  }
  curandDestroyGenerator(gen);
}

} // namespace cuda_example