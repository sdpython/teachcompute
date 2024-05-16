#include "teachcompute_helpers.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace cuda_example {

enum TransType { FalseFalse = 0, FalseTrue = 1, TrueFalse = 2, TrueTrue = 3 };

#define BLOCK_SIZE 32
#define CEIL_DIV(N, DEN) ((N + (DEN - 1)) / DEN)

template <typename T, TransType trans_type>
__global__ void kernel_matmul_v1(int M, int N, int K, const T *A, const T *B,
                                 T *C) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (trans_type == TransType::FalseFalse) {
    if (x < M && y < N) {
      T tmp = 0.0;
      for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
      }
      C[x * N + y] += tmp;
    }
  }
  else if (trans_type == TransType::TrueFalse) {
    if (x < M && y < N) {
      T tmp = 0.0;
      for (int i = 0; i < K; ++i) {
        tmp += A[x * M + i] * B[i * N + y];
      }
      C[x * N + y] += tmp;
    }
  }
}

template <typename T>
int _matmul_v1(int n_rows1, int n_cols1, const T *A, int n_rows2, int n_cols2,
               const T *B, T *C, bool transA, bool transB) {
  int M, N, K;
  TransType tt;
  if (transA) {
    if (transB) {
      tt = TransType::TrueTrue;
    } else {
      tt = TransType::TrueFalse;
      EXT_ENFORCE(n_rows1 == n_rows2, "Dimensions do not match.");
      M = n_cols1;
      N = n_cols2;
      K = n_rows1;
    }
  } else {
    if (transB) {
      tt = TransType::FalseTrue;
    } else {
      tt = TransType::FalseFalse;
      EXT_ENFORCE(n_cols1 == n_rows2, "Dimensions do not match.");
      M = n_rows1;
      N = n_cols2;
      K = n_cols1;
    }
  }
  dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE), 1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  switch (tt) {
  case TransType::FalseFalse:
    kernel_matmul_v1<T, TransType::FalseFalse>
        <<<gridDim, blockDim>>>(M, N, K, A, B, C);
    break;
  case TransType::TrueFalse:
    kernel_matmul_v1<T, TransType::TrueFalse>
        <<<gridDim, blockDim>>>(M, N, K, A, B, C);
    break;
  default:
    EXT_THROW("Not implemented yet for trans*=", (int)tt, ".");
  }
  cudaDeviceSynchronize();
  return K;
}

int matmul_v1(int n_rows1, int n_cols1, const float *A, int n_rows2,
              int n_cols2, const float *B, float *C, bool transA, bool transB) {
  return _matmul_v1(n_rows1, n_cols1, A, n_rows2, n_cols2, B, C, transA,
                    transB);
}

} // namespace cuda_example
